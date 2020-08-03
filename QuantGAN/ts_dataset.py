import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import os


class TsDataset(Dataset):
    """Time Series dataset."""

    def __init__(self, csv_file, csv_param, seq_length, bfill = False, normalize = True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            csv_param (string): Path to the csv file containing parameters.
            seq_length (int): Length of subsequence to create
            bfill (Boolean, optional): Whether to backfill or forward fill the data.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # the original dataframe
        #By default, the input is not normalized to [-1, 1]
        self.isNormalized = False

        self.frame = pd.read_csv(csv_file)
        self.seq_length = seq_length
        ndata = self.frame.to_numpy()


        df_index = ndata[:, 0]
        date = ndata[:, 1]
        ndata = ndata[:, 2]

        self.Data_max = ndata.max()
        self.Data_min = ndata.min()

        #convert to a torch double
        #ndata = torch.from_numpy(ndata).double()
        #call normalize
        #ndata = self.normalize(ndata) if normalize else ndata
        #convert back to numpy
        #ndata = ndata.data.numpy()

        # the one is for the 1th channel
        newdata = np.zeros(shape=(len(ndata) - seq_length, 1, seq_length), dtype=float)

        ## forward_fill (non-causal, paper definition)
        if (not bfill):
            for i in range(0, len(self.frame) - seq_length):
                newdata[i, 0, :] = np.ravel(ndata[i:i + seq_length])
        ## backward fill (causal)
        else:
            for i in range(seq_length, len(self.frame)):
                newdata[i - seq_length, 0, :] = np.ravel(ndata[i - seq_length:i])
        # can be converted back into a dataframe by using df_index
        self.Data = (df_index[:-seq_length], date[:-seq_length], newdata)

        if(normalize):
            self.normalize()

        self.paramFrame = pd.read_csv(csv_param)
        self.params = self.paramFrame.to_numpy()
        self.params = np.ravel(self.params)
        #self.params = np.double(self.params)
        #params0 is delta, 1 and 2 is mean / sd of regular series, 3 4 is params of inv lam transformed series

    def __len__(self):
        return len(self.Data[0])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.Data[0][idx], self.Data[1][idx], self.Data[2][idx])


    def normalize(self):
        """Normalize this object in [-1,1] range, saving statics for denormalization"""
        temp = (2 * (self.Data[2] - self.Data_min) / (self.Data_max - self.Data_min) - 1)
        self.Data = (self.Data[0], self.Data[1], temp)
        self.isNormalized = True

    def denormalize(self):
        """Revert [-1,1] normalization"""

        #if not hasattr(self, 'max') or not hasattr(self, 'min'):
        #    raise Exception("You are calling denormalize, but the input was not normalized")
        if( self.isNormalized == False):
            raise Exception("You cannot call denormalize when the data is not normalized.")
        temp = 0.5 * (self.Data[2] * self.Data_max - self.Data[2] * self.Data_min + self.Data_max + self.Data_min)
        self.Data = (self.Data[0], self.Data[1], temp)
        self.isNormalized = False

    def write(self, idx, values, denormalize = True):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if torch.is_tensor(values):
            values = values.data.numpy()

        for i in range(0, len(idx)):
            self.Data[2][idx[i]] = values[i, 0, :]

    def to_CSV(self, path, rn):
        name = "GeneratedData" + rn + ".csv"
        myfile = os.path.join(path, name)
        new_frame = pd.DataFrame(np.column_stack( ( self.Data[0], self.Data[1] )), columns = ['Index', 'Date'] )

        for i in range(0,  self.seq_length):
            mystr = "T_Index" + str(i)
            new_frame[mystr] = self.Data[2][:, 0, i]
        new_frame.to_csv(myfile, index = False)

    # 9. Recreate the data (can be done in Python with parameters file)
    # Recreated 2 is original log returns
    # Recreated 1 is the zero mean, unit variance log returns (normalized log returns)

    #Recreated1 = (lret.SP500.InvLam.Norm * exp(d / 2 * lret.SP500.InvLam.Norm ^ 2)) * sd(lret.SP500.InvLam) + mean(
    #    lret.SP500.InvLam)
    #Recreated2 = Recreated1 * sd(lret.SP500) + mean(lret.SP500)

    def rescale(self):
        """Revert Mean/Var scaling """

        #=E3*Param!$E$2+Param!$D$2
        #=F2 * Param!$C$2 + Param!$B$2
        #temp = self.Data[2] * self.params[4] + self.params[3]
        temp = self.Data[2]*self.params[2] + self.params[1]
        self.Data = (self.Data[0], self.Data[1], temp)
        self.isNormalized = False

    def lambert(self):
        """Revert Inverse Lambert normalization"""
        #=D2 * EXP(Param!$A$2 / 2 * POWER(D2, 2))
        temp = self.Data[2]*np.exp( self.params[0]/2 * np.power(self.Data[2], 2))
        temp = temp * self.params[4] + self.params[3]
        self.Data = (self.Data[0], self.Data[1], temp)