from pathlib import Path
import pandas as pd
import numpy as np
from model.data_manip import get_sep
from main import run
from model.utils import filter_profiles_with_extreme_peaks, split_data_train_test


####################################################################################################

# Model type ('GAN' or 'WGAN')
MODEL_TYPE = 'WGAN'

# ──────────────────────────────────────────────────────────────────────────────────────────

# Project name
PROJECT_NAME = 'project_1'

# Input file path
INPUT_PATH = Path.cwd() / 'data' / ...

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FORMAT = '.npy'

# Log stats
LOG_STATS = True

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = False

# Set the number of epochs
EPOCH_COUNT = 5_000

# Change the result save frequency; save all samples/models in addition to visualizations
SAVE_FREQ = 50000000
SAVE_MODELS = True
SAVE_PLOTS = True
SAVE_SAMPLES = True
CHECK_FOR_MIN_STATS = 10   #after these epochs, runs with lower stats than the current minimum are plotted

####################################################################################################

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = None #Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/GAN/runs/Enercoop_WGAN_80_20/lucky-wood-392_2025-05-21-111356060/models/epoch_95") / "epoch_95.pt"

# Create synthetic data from existing model (if True, there is no training)
CREATE_DATA = False

# ==========================================================================================
#                                        MAIN SCRIPT
# ==========================================================================================

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
    params['checkForMinStats'] = CHECK_FOR_MIN_STATS
    inputFile = pd.read_csv(INPUT_PATH, sep = get_sep(INPUT_PATH))
    inputFile = inputFile.set_index(inputFile.columns[0])
    inputFile.index = pd.to_datetime(inputFile.index, format = 'mixed')
    if np.isnan(inputFile).any().any():
        print("Warning: NaN values detected in input file")
        exit()
    # Filter out profiles with maximum values within 95th percentile
    filtered_inputFile = filter_profiles_with_extreme_peaks(inputFile)

    train_data, test_data = split_data_train_test(filtered_inputFile, train_ratio=0.8, random_state=42)
    
    run(params, MODEL_TYPE, PROJECT_NAME, train_data=train_data, LOG_STATS=LOG_STATS, USE_WANDB=USE_WANDB, MODEL_PATH=MODEL_PATH, CREATE_DATA=CREATE_DATA, generate_n_profiles = 100, test_data=test_data)