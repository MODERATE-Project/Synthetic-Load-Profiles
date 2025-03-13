from pathlib import Path
from model.params import params
import pandas as pd
from model.data_manip import get_sep
from main import run

####################################################################################################

# Project name
PROJECT_NAME = 'test_continue'

# Input file path
INPUT_PATH = Path.cwd() / 'data' / 'smart_meters_london_resampled.csv'

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FORMAT = '.npy'

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = True

# Set the number of epochs
EPOCH_COUNT = 100

# Change the result save frequency; save all samples/models in addition to visualizations
SAVE_FREQ = 50
SAVE_MODELS = False
SAVE_PLOTS = True
SAVE_SAMPLES = True

####################################################################################################

# Model state path (optional, for continuation of training or generation of data)
#MODEL_PATH = r'C:/Users/Arbeit/Projekte/Git/GAN/runs/test_continue/chocolate-lion-220_2025-03-13-123914890/models/epoch_100/epoch_100.pt.gz'
MODEL_PATH = None

# Create synthetic data from existing model (if True, there is no training)
CREATE_DATA = False

####################################################################################################

if __name__ == '__main__':
    params['outputFormat'] = OUTPUT_FORMAT
    params['epochCount'] = EPOCH_COUNT
    params['saveFreq'] = SAVE_FREQ
    params['saveModels'] = SAVE_MODELS
    params['savePlots'] = SAVE_PLOTS
    params['saveSamples'] = SAVE_SAMPLES
    inputFile = pd.read_csv(INPUT_PATH, sep = get_sep(INPUT_PATH))
    inputFile = inputFile.set_index(inputFile.columns[0])
    run(params, PROJECT_NAME, inputFile, USE_WANDB, MODEL_PATH, CREATE_DATA)