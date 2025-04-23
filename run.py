from pathlib import Path
import pandas as pd
from model.data_manip import get_sep
from main import run

# ==========================================================================================
#                                        PARAMETERS
# ==========================================================================================

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
EPOCH_COUNT = 100

# Change the result save frequency; save all samples/models in addition to visualizations
SAVE_FREQ = 10
SAVE_MODELS = False
SAVE_PLOTS = True
SAVE_SAMPLES = False

# Check if metric improves starting with the specified epoch and only plot "better" results
CHECK_FOR_MIN_STATS = 100

# ──────────────────────────────────────────────────────────────────────────────────────────

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = None

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
    run(
        params = params,
        modelType = MODEL_TYPE,
        projectName = PROJECT_NAME,
        inputFile = inputFile,
        logStats = LOG_STATS,
        useWandb = USE_WANDB,
        modelPath = MODEL_PATH,
        createData = CREATE_DATA
    )