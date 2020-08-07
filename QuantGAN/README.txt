## Introduction

The QuantGAN paper describes a deep learning model that uses a GAN with temporal convolutional networks (TCNs) as the generator and discriminator to simulate stock
price evolutions. The primary objective for the GAN architecture is to outperform standard SDE models, and the specific architecture of the model presented in the paper
should be a competitive alternative to traditional Monte Carlo methods used for modeldriven methods (e.g. Heston) and also for historical simulation.

Consider a standard geometric Brownian motion SDE. There are two immediately apparent problems. First, the Brownian motion increments are assumed to be normally
distributed and independent. Both of these assumptions are untrue. It is known that market returns are non- normal and instead heavy tailed, meaning the increments should
not be normally distributed. Second, the increment on day 2 is dependent on the increment of day 1. These increments are not uncorrelated in the market.

Thus, the GAN architecture should do a better job capturing the effects of volatility clustering and other phenomenon observed in real stock return data, as opposed to
simulated data. The mathematics of the construction of these models is discussed in great detail in the paper. The work done for this project involves the following. First, a preprocessing routine was created to transform the data into an acceptable input for the model. Second, an
extensible framework was created to utilize a variety of architectures for training noise.

Third, a seamless integration of input and output data with the model. All code was handwritten, with the exception of the TCNBlock code, which according
to this paper and the paper it references has a very specific format. The time series data set code was written from scratch, the main.py was modeled after some links we found
online to help, and the MLP and LSTM models were also written completely from scratch. Besides the integrated models implemented in Pytorch, we also implemented
the MLP-GAN and LSTM-GAN in TensorFlow. You can find the codes under the “/Tensorflow_implementation”.

## Steps to Reproduce the Project

1. Download the ZIP file to work locally, or add it to your GitHub repository.

2. Verify that all requirements are met from the requirements.txt folder. If you use PyCharm as an IDE, and load in the files as a PyCharm project, PyCharm should
automatically tell you if you are unable to meet any of the requirements.

3. Run Preprocessing.R. This script currently is hardcoded to transform a file “GSPC.csv” which is the name of the SP500 data downloaded from Yahoo
finance. Any csv file from Yahoo finance can take the place of SP500 if one wishes to run the script on other data.

4. In command prompt or in terminal, navigate to the directory in which you have
stored this project.

5. Type “python main.py.” There are additional arguments that can be passed in,
and this will be discussed below in detail.

6. When the script is finished running (Roughly 40 minutes for CPU execution on a 4C/8T system, faster for GPU or cloud computation), it will automatically
store results in the “out/X/” directory, where X should show the date and time (Hours_minutes) when you ran this project. The name of the folder can also be
modified, and is explained below.
7. Open up “GeneratedData.csv” to view the resultant time series.
