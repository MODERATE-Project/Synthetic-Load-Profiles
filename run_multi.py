from pathlib import Path
import pandas as pd
from model.data_manip import get_sep
from main import run

####################################################################################################

# Model type ('GAN' or 'WGAN')
MODEL_TYPE = 'GAN'

####################################################################################################

# Project name
PROJECT_NAME = 'GAN_VITO'

# Input file path
INPUT_PATH = Path.cwd() / 'data' / 'Consumption_data_hourly.csv'

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FORMAT = '.npy'

# Log RMSE
LOG_RMSE = True

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = False

# Set the number of epochs
EPOCH_COUNT = 400

# Change the result save frequency; save all samples/models in addition to visualizations
SAVE_FREQ = 20
SAVE_MODELS = False
SAVE_PLOTS = True
SAVE_SAMPLES = True

####################################################################################################

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = None

# Create synthetic data from existing model (if True, there is no training)
CREATE_DATA = False

####################################################################################################

# Import labels for splitting dataframe
df_label = pd.read_csv(Path.cwd() / 'data' / 'Labels_consumption_data.csv', sep = ';')

####################################################################################################

if __name__ == '__main__':
    if MODEL_TYPE == 'GAN':
        from model.GAN_params import params
    elif MODEL_TYPE == 'WGAN':
        from model.WGAN_params import params
    params['outputFormat'] = OUTPUT_FORMAT
    params['epochCount'] = EPOCH_COUNT
    params['saveFreq'] = SAVE_FREQ
    params['saveModels'] = SAVE_MODELS
    params['savePlots'] = SAVE_PLOTS
    params['saveSamples'] = SAVE_SAMPLES
    inputFile = pd.read_csv(INPUT_PATH, sep = get_sep(INPUT_PATH))
    inputFile = inputFile.set_index(inputFile.columns[0])
    inputFile.index = pd.to_datetime(inputFile.index, format = 'mixed')
    for cat in df_label['Category'].unique():
        IDs = [str(x) for x in df_label.loc[df_label['Category'] == cat, 'ID']]
        df = inputFile.loc[:, IDs]
        run(params, MODEL_TYPE, f"{PROJECT_NAME}_{cat}", df, LOG_RMSE, USE_WANDB, MODEL_PATH, CREATE_DATA)