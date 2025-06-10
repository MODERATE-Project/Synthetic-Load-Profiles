from pathlib import Path
import pandas as pd
import numpy as np
from model.data_manip import get_sep
from main import run

def split_data_train_test(input_file, labels_file=None, train_ratio=0.8, random_state=42):
    """
    Split the data into training and test sets. If labels are provided, maintains category proportions.
    Otherwise performs a random split.
    
    Args:
        input_file (pd.DataFrame): The main data file with time series
        labels_file (pd.DataFrame, optional): The file containing EAN_ID and label information
        train_ratio (float): Ratio of training data (default: 0.8)
        random_state (int): Random seed for reproducibility
        
    Returns:
        tuple: (train_data, test_data) as DataFrames
    """
    np.random.seed(random_state)
    
    # If no labels provided, perform simple random split
    if labels_file is None:
        n_columns = len(input_file.columns)
        n_train = int(n_columns * train_ratio)
        
        # Randomly select columns for train and test
        train_columns = np.random.choice(input_file.columns, n_train, replace=False)
        test_columns = list(set(input_file.columns) - set(train_columns))
        
        # Create train and test datasets
        train_data = input_file[train_columns]
        test_data = input_file[test_columns]
        
        # Print statistics
        print("\nData split statistics:")
        print(f"Total number of columns: {n_columns}")
        print(f"Training set size: {len(train_columns)} columns")
        print(f"Test set size: {len(test_columns)} columns")
        
        return train_data, test_data
    
    # If labels are provided, maintain category proportions
    # Create a mapping of EAN_ID to category
    id_to_category = dict(zip(labels_file['EAN_ID'], labels_file['label']))
    
    # Get all unique categories
    categories = labels_file['label'].unique()
    
    # Initialize empty lists for train and test columns
    train_columns = []
    test_columns = []
    
    # For each category, split the corresponding columns
    for category in categories:
        # Get all EAN_IDs for this category
        category_ids = labels_file[labels_file['label'] == category]['EAN_ID']
        
        # Get the corresponding columns from input_file
        category_columns = [col for col in input_file.columns if int(col) in category_ids.values]
        
        # Calculate split size
        n_columns = len(category_columns)
        n_train = int(n_columns * train_ratio)
        
        # Randomly select columns for train and test
        train_cols = np.random.choice(category_columns, n_train, replace=False)
        test_cols = list(set(category_columns) - set(train_cols))
        
        train_columns.extend(train_cols)
        test_columns.extend(test_cols)
    
    # Create train and test datasets
    train_data = input_file[train_columns]
    test_data = input_file[test_columns]
    
    # Print some statistics
    print("\nData split statistics:")
    print(f"Total number of columns: {len(input_file.columns)}")
    print(f"Training set size: {len(train_columns)} columns")
    print(f"Test set size: {len(test_columns)} columns")
    
    # Print category distribution
    print("\nCategory distribution in training set:")
    train_categories = [id_to_category[int(col)] for col in train_columns]
    print(pd.Series(train_categories).value_counts())
    
    print("\nCategory distribution in test set:")
    test_categories = [id_to_category[int(col)] for col in test_columns]
    print(pd.Series(test_categories).value_counts())
    
    return train_data, test_data

def filter_profiles_with_extreme_peaks(df, percentile=95):
    """
    Filter out columns where the maximum value (peak) is within the specified percentile of all values.
    
    Args:
        df (pd.DataFrame): Input DataFrame with time series data
        percentile (float): Percentile threshold (default: 95)
        
    Returns:
        pd.DataFrame: Filtered DataFrame with only columns that have maximum values above the percentile
    """
    # Calculate maximum value for each column
    max_values = df.max()
    
    # Calculate the percentile threshold of all values in the dataset
    percentile_threshold = np.percentile(max_values.values.flatten(), 95)
    
    # Filter columns where maximum value is above the percentile threshold
    filtered_columns = max_values[max_values < percentile_threshold].index
    
    # Return filtered DataFrame
    return df[filtered_columns]

####################################################################################################

# Model type ('GAN' or 'WGAN')
MODEL_TYPE = 'WGAN'

####################################################################################################

# Project name
PROJECT_NAME = 'London_WGAN_80_20_kicked_out_max_95_percentile'

# Input file path
# INPUT_PATH = Path.cwd() / 'data' / "raw_data" / "enercoop" / "ENERCOOP_1year_filtered.csv"
# INPUT_PATH = Path.cwd() / "data" / "Fluvius_processed" /'fluvius_wide_format.csv'
INPUT_PATH = Path.cwd() / "data" / "smart_meters_london_resampled.csv" 

# Output file format ('npy', 'csv' or 'xlsx')
OUTPUT_FORMAT = '.npy'

# Log RMSE
LOG_STATS = True

# Use Wandb (if True, metric will be tracked online; Wandb account required)
USE_WANDB = True

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