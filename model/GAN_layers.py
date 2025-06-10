from torch import nn
from collections import OrderedDict

from model.GAN_params import params


DIM_NOISE = params['dimNoise']
DIM_HIDDEN = params['dimHidden']
CHANNEL_COUNT = params['channelCount']
P_DROPOUT = params['dropout']

layersGen = nn.Sequential(OrderedDict([
    ('1', nn.Sequential(
        nn.ConvTranspose2d(in_channels = DIM_NOISE, out_channels = 8*DIM_HIDDEN, kernel_size = (24, 22), stride = (1, 1), padding = (0, 0), bias = False),
        nn.BatchNorm2d(num_features = 8*DIM_HIDDEN),
        nn.ReLU(inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('2', nn.Sequential(
        nn.ConvTranspose2d(in_channels = 8*DIM_HIDDEN, out_channels = 4*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
        nn.BatchNorm2d(num_features = 4*DIM_HIDDEN),
        nn.ReLU(inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('3', nn.Sequential(
        nn.ConvTranspose2d(in_channels = 4*DIM_HIDDEN, out_channels = 2*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
        nn.BatchNorm2d(num_features = 2*DIM_HIDDEN),
        nn.ReLU(inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('4', nn.Sequential(
        nn.ConvTranspose2d(in_channels = 2*DIM_HIDDEN, out_channels = DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
        nn.BatchNorm2d(num_features = DIM_HIDDEN),
        nn.ReLU(inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('out', nn.Sequential(
        nn.ConvTranspose2d(in_channels = DIM_HIDDEN, out_channels = CHANNEL_COUNT, kernel_size = (3, 4), stride = (1, 2), padding = (1, 0), bias = False),
        nn.Tanh()
    ))
]))

layersDis = nn.Sequential(OrderedDict([
    ('1', nn.Sequential(
        nn.Conv2d(in_channels = CHANNEL_COUNT, out_channels = DIM_HIDDEN, kernel_size = (3, 4), stride = (1, 2), padding = (1, 0), bias = False),
        nn.LeakyReLU(negative_slope = 0.2, inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('2', nn.Sequential(
        nn.Conv2d(in_channels = DIM_HIDDEN, out_channels = 2*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
        nn.BatchNorm2d(num_features = 2*DIM_HIDDEN),
        nn.LeakyReLU(negative_slope = 0.2, inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('3', nn.Sequential(
        nn.Conv2d(in_channels = 2*DIM_HIDDEN, out_channels = 4*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
        nn.BatchNorm2d(num_features = 4*DIM_HIDDEN),
        nn.LeakyReLU(negative_slope = 0.2, inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('4', nn.Sequential(
        nn.Conv2d(in_channels = 4*DIM_HIDDEN, out_channels = 8*DIM_HIDDEN, kernel_size = (3, 3), stride = (1, 2), padding = (1, 0), bias = False),
        nn.BatchNorm2d(num_features = 8*DIM_HIDDEN),
        nn.LeakyReLU(negative_slope = 0.2, inplace = True),
        nn.Dropout2d(p = P_DROPOUT)
    )),
    ('out', nn.Sequential(
        nn.Conv2d(in_channels = 8*DIM_HIDDEN, out_channels = 1, kernel_size = (24, 22), stride = (1, 1), padding = (0, 0), bias = False),
        nn.Sigmoid()
    ))
]))