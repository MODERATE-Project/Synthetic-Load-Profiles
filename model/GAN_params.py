import torch
from torch import nn


# GAN parameters and their default values
params = {
    'batchSize': 40,
    'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    'lossFct': nn.BCELoss(),
    'lrGen': 1e-4/3.25,
    'lrDis': 1e-4/2.25,
    'betas': (0.5, 0.999),
    'epochCount': 250,
    'labelReal': 1,
    'labelFake': 0,
    'saveFreq': 1000,
    'genLoopCount': 1,
    'saveModels': False,
    'savePlots': True,
    'saveSamples': False,
    'checkForMinStats': 100,
    'dimNoise': 128,
    'dimHidden': 16,
    'channelCount': 1,
    'outputFormat': '.npy',
    'dropoutOffEpoch': 1000,
    'dropout': 0.2
}