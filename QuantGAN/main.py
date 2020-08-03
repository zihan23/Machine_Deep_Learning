#!/usr/bin/python
import time
startTime = time.time()
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import datetime
from ts_dataset import TsDataset
from tensorboardX import SummaryWriter
from models.recurrent_models import LSTMGenerator, LSTMDiscriminator
from models.convolutional_models import CausalConvGenerator, CausalConvDiscriminator
from models.mlp_models import MLPGenerator, MLPDiscriminator

parser = argparse.ArgumentParser()

#Added
parser.add_argument('--backfill', default="False", help='Create (real) time series by looking at the previous values')
parser.add_argument('--num_layers', default="4", help='Number of total layers (Tblocks) to include in the model')
parser.add_argument('--terminal_cut', default="True", help='Take endpoint of subsequence for BCELoss')
parser.add_argument('--parameters_path', default="Parameters_GSPC.csv", help='path to parameters of dataset')
parser.add_argument('--debug', default=1, help='ECHO Loss. Not recommended.')
parser.add_argument('--weight_decay', default = 0, help='weight decay for adam optimization.')
parser.add_argument('--dataset', default="ts", help='dataset to use (only ts for now)')
parser.add_argument('--dataset_path', default="Normalized_GSPC.csv", help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers. Large values can lead to instability.', default=2)
parser.add_argument('--batchSize', type=int, default=4, help='input batch size')
parser.add_argument('--nz', type=int, default=1, help='dimensionality of the latent vector z (num channels)')
parser.add_argument('--epochs', type=int, default=100, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='checkpoints', help='folder to save checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
#parser.add_argument('--logdir', default='log', help='logdir for tensorboard')
parser.add_argument('--outdir', default='out', help='outdir for trajectory')
parser.add_argument('--run_tag', default='', help='tags for the current run')
parser.add_argument('--checkpoint_every', default=5, help='number of epochs after which saving checkpoints') 
parser.add_argument('--dis_type', default='tcn', choices=['tcn','lstm','mlp'], help='architecture to be used for discriminator to use')
parser.add_argument('--gen_type', default='tcn', choices=['tcn','lstm','mlp'], help='architecture to be used for generator to use')
opt = parser.parse_args()

try:
    os.makedirs(opt.outf)
except OSError:
    pass


if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("You have a CUDA device, but have set the arguments to not run with CUDA. "
          "Consider running with --cuda enabled for better performance.")


subseqlen = 2**int(opt.num_layers)
dataset = TsDataset(opt.dataset_path, opt.parameters_path, subseqlen, opt.backfill)
assert dataset

fakeOutputTS = TsDataset(opt.dataset_path, opt.parameters_path, subseqlen, opt.backfill)
assert fakeOutputTS

dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers), drop_last = True)

device = torch.device("cuda:0" if opt.cuda else "cpu")
nz = int(opt.nz)
#Retrieve the sequence length as first dimension of a sequence in the dataset


if opt.dataset == "ts":
    total_seq_len = len(dataset)


#An additional input is needed for the delta
#opt.nz should be 1 if only one time series
in_dim = opt.nz
if opt.dis_type == "lstm":
    netD = LSTMDiscriminator(in_dim=in_dim, subseqlen=subseqlen, hidden_dim=5 ).to(device)
elif opt.dis_type == "tcn":
    netD = CausalConvDiscriminator(input_size=in_dim, n_layers=4, n_channel=1, kernel_size=2, dropout=0.1).to(device)
elif opt.dis_type == "mlp":
    netD = MLPDiscriminator(input_size = in_dim, n_layers=4, n_channel =1).to(device)

if opt.gen_type == "lstm":
    netG = LSTMGenerator(in_dim=in_dim, out_dim=1, subseqlen = subseqlen, hidden_dim=5,).to(device)
elif opt.gen_type == "tcn":
    netG = CausalConvGenerator(noise_size=in_dim, output_size=1, n_layers=4, n_channel=1, kernel_size=2, dropout=0.1).to(device)
elif opt.gen_type == "mlp":
    netG = MLPGenerator(noise_size = in_dim, output_size = 1, n_layers=4, n_channel = 1).to(device)


assert netG
assert netD

#This is to reuse already created generators and discriminators
if opt.netG != '':
    netG.load_state_dict(torch.load(opt.netG))    
if opt.netD != '':
    netD.load_state_dict(torch.load(opt.netD))

#print("|Discriminator Architecture|\n", netD)
#print("|Generator Architecture|\n", netG)

criterion = nn.BCELoss().to(device)
real_label = 1
fake_label = 0

# setup optimizer
optimizerD = optim.Adam(netD.parameters(), lr=opt.lr, weight_decay = float(opt.weight_decay))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, weight_decay = float(opt.weight_decay))

netG = netG.double()
netD = netD.double()




def main():
    date = datetime.datetime.now().strftime("%m-%d-%y_%H_%M")
    run_name = f"{opt.run_tag}_{date}" if opt.run_tag != '' else date
    out_dir_name = os.path.join(opt.outdir, run_name)
    if not os.path.exists(out_dir_name):
        os.makedirs(out_dir_name)
    path = os.path.join(out_dir_name, "output.txt")
    OutTraj = open(path, "w")
    OutTraj.write(str(opt))
    OutTraj.write("\n")
    for epoch in range(opt.epochs):
        for i, (idx, date, data) in enumerate(dataloader, 0):
            # data is of shape batch size by subsequence length
            niter = epoch * len(dataloader) + i

            '''
            reshaping code:
            attempt to force data into the proper dimensionality for conv1d
            data = data.reshape(data.shape[0], 1, data.shape[1])
            data = data.double()        
            '''


            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################

            #Train with real data
            netD.zero_grad()
            real = data.to(device)
            #
            # # THIS IS THE SIZE OF NOISE, REAL DATA, FAKE DATA that is INPUT to the Discriminator and Generator
            # # GENERATOR OUTPUT (fake data) MUST ALSO BE THIS SIZE
            # # It is OK if Discriminator output is not this size.
            # # Discriminator output size at MINIMUM must be a matrix of batch_size, num_channels, 1
            #
            batch_size, num_channels, seq_len = real.size(0), real.size(1), real.size(2)
            #Should the entire subsequence be given a label of 1, or should each point in the subsequence be given a label of 1
            # Intuition: The entire subsequence should be given a label of 1.

            #label = torch.full((batch_size, num_channels, seq_len), real_label, device=device)
            label = torch.full((batch_size, num_channels, 1), real_label, device=device)

            label = label.double()
            real = real.double()
            output = netD(real)
            #this output is a subsequence.
            output = output.double()

            #If we create label as batch_size, num_channels, 1, take only terminal value of output
            # I don't think this is how we should do things
            if(opt.terminal_cut):
                output = output[:, :, -1]
                #mean of all 16
                #non-uniform weighted mean

            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()

            #Train with fake data
            #noise = torch.randn(batch_size, seq_len, nz, device=device)
            noise = torch.randn(batch_size, nz, seq_len, device = device)

            noise = noise.double()
            #noise = noise.float()

            fake = netG(noise)
            fake = fake.double()
            fakeOutputTS.write( idx, fake )

            #fake = fake.float()

            label.fill_(fake_label)
            output = netD(fake.detach())
            output = output.double()
            #output = output.float()

            if(opt.terminal_cut):
                output = output[:, :, -1]

            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()
            label.fill_(real_label)
            output = netD(fake)
            #output = output.float()
            output = output.double()

            if(opt.terminal_cut):
                output = output[:, :, -1]

            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()

            optimizerG.step()

            ###########################
            # (3) Supervised update of G network: minimize mse of input deltas and actual deltas of generated sequences
            ###########################

            #Report Metrics/Output Trajectory
            OutString = ('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, opt.epochs, i, len(dataloader), errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            OutTraj.write(OutString)
            OutTraj.write('\n')

            if(opt.debug):
                print(OutString, end='')
                print('')

        ##### End of the epoch #####
        print("Epoch %d completed." % epoch)
        # Checkpoint
        # Creating a checkpoint throws a ton of warnings
        if (epoch % opt.checkpoint_every == 0) or (epoch == (opt.epochs - 1)):
            torch.save(netG, '%s/%s_netG_epoch_%d.pth' % (opt.outf, opt.run_tag, epoch))
            torch.save(netD, '%s/%s_netD_epoch_%d.pth' % (opt.outf, opt.run_tag, epoch))

    #denormdata = fakeOutputTS.denormalize(fakeOutputTS.Data[2])
    #fakeOutputTS = fakeOutputTS.writeNewData( denormdata )

    fakeOutputTS.denormalize()
    fakeOutputTS.lambert()
    fakeOutputTS.rescale()
    fakeOutputTS.to_CSV(out_dir_name, run_name)
    endTime = time.time()
    value = endTime - startTime
    str3 = ('Total computation time is %.4f seconds.' % value)
    print(str3)
    OutTraj.write(str3)
    OutTraj.close()
    #outpath = os.path.join(out_dir_name, "output.txt")
    #os.rename("output.txt", outpath)
## Windows multiprocessing protection
if __name__ == '__main__':
    main()