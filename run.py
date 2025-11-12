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
PROJECT_NAME = 'WGAN_final_all'

# Input file path
INPUT_PATH = Path.cwd() / 'data' / "sample_data" / "opendata_fluvius" / "fluvius_wide_format.csv"

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FORMAT = '.npy'

# Log stats
LOG_STATS = True

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = True

# Set the number of epochs
EPOCH_COUNT = 10_000

# Change the result save frequency; save all samples/models in addition to visualizations
SAVE_FREQ = 50000000
SAVE_MODELS = True
SAVE_PLOTS = True
SAVE_SAMPLES = True
CHECK_FOR_MIN_STATS = 10   #after these epochs, runs with lower stats than the current minimum are plotted

####################################################################################################

# Model state path (optional, for continuation of training or generation of data)
MODEL_PATH = Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Synthetic-Load-Profiles/runs/WGAN_final_all/fearsome-omen-403_2025-10-31-135650489/models/epoch_899") / "epoch_899.pt"

create_data_paths = [
    Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Synthetic-Load-Profiles/runs/WGAN_final_EV/shadowed-coffin-406_2025-10-31-144954467/models/epoch_20920") / "epoch_20920.pt",
    Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Synthetic-Load-Profiles/runs/WGAN_final_EV+PV/mystical-raven-414_2025-10-31-195113535/models/epoch_2691") / "epoch_2691.pt",
    Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Synthetic-Load-Profiles/runs/WGAN_final_heat pump+PV/ghastly-ripple-413_2025-10-31-181921063/models/epoch_4695") / "epoch_4695.pt",
    Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Synthetic-Load-Profiles/runs/WGAN_final_PV/frightful-echo-408_2025-10-31-163454888/models/epoch_5872") / "epoch_5872.pt",
    Path(r"/home/users/pmascherbauer/projects4/workspace_philippm/Synthetic-Load-Profiles/runs/WGAN_final_standard/murky-charm-405_2025-10-31-144919919/models/epoch_5748") / "epoch_5748.pt",

]

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

    labels_file = pd.read_csv(Path.cwd() / 'data' / "sample_data" / "opendata_fluvius" / "fluvius_indicators.csv")
    if np.isnan(inputFile).any().any():
        print("Warning: NaN values detected in input file")
        exit()
    # Filter out profiles with maximum values within 95th percentile
    # inputFile = filter_profiles_with_extreme_peaks(inputFile)

    # for training on subgroups of the data
    # for name in labels_file["label"].unique():
    #     PROJECT_NAME = f'WGAN_final_{name}'
    #     name_id = name.replace("WGAN_final_", "")
    #     ids = labels_file[labels_file["label"] == name]["EAN_ID"].to_list()
    #     data = inputFile.loc[:, [str(i) for i in ids]]

    #     train_data, test_data = split_data_train_test(data, train_ratio=0.8, random_state=42)

    #     run(params, MODEL_TYPE, PROJECT_NAME, 
    #         train_data=train_data, 
    #         logStats=LOG_STATS, 
    #         useWandb=USE_WANDB, 
    #         modelPath=create_data_paths[0], 
    #         createData=CREATE_DATA, 
    #         useMarimo=False,
    #         generate_n_profiles = 1300, 
    #         test_data=test_data
    #         )
    #     break

    # train full model on all data
    train_data, test_data = split_data_train_test(inputFile, labels_file=labels_file, train_ratio=0.8, random_state=42)
    run(params, MODEL_TYPE, PROJECT_NAME, 
        train_data=train_data, 
        logStats=LOG_STATS, 
        useWandb=USE_WANDB, 
        modelPath=MODEL_PATH, 
        createData=True, 
        useMarimo=False,
        generate_n_profiles = 50_000, 
        test_data=test_data)
    
    # generate differet data sizes from each submodel (trained on the different subgroups and on the full data)
    # for generation_nr in [100, 300, 1000, 5000, 50_000, 100_000]:
    #     for model_path in create_data_paths:
    #         run(params, 
    #             MODEL_TYPE, 
    #             PROJECT_NAME, 
    #             train_data=train_data, 
    #             logStats=LOG_STATS, 
    #             useWandb=USE_WANDB, 
    #             modelPath=model_path, 
    #             createData=True, 
    #             useMarimo=False,
    #             generate_n_profiles = generation_nr, 
    #             test_data=test_data)
    