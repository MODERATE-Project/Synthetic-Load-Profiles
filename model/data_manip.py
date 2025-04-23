import pandas as pd
import numpy as np
import csv
import pathlib


def add_filler_rows(df: pd.DataFrame, dayCount: int) -> pd.DataFrame:
    """
    Appends additional rows to the input dataframe in the case that the number of days
    (which equals the number of rows of `df` divided by 24) is lower than `dayCount`.

    Args:
        df (pd.DataFrame): A dataframe containing multiple consumption profiles. Each column
            should correspond to one profile and contain hourly values for one year.

        dayCount (int): The (maximum) number of days to be processed by the GAN. If this
            value is changed, the layer structure of the Generator and the Discriminator
            need to be adapted accordingly.

    Raises:
        ValueError: An error is produced if the number of days exceeds `dayCount` or is less
        than 365.

    Returns:
        pd.DataFrame: An updated dataframe with additional filler rows.
    """
    addRowCount = 24 * dayCount - df.shape[0]
    if addRowCount < 0:
        raise ValueError(
            f"The maximum amount of days allowed is {dayCount} (= {dayCount * 24} rows)!"
        )
    maxAddRowCount = (dayCount - 365) * 24
    if addRowCount > maxAddRowCount:
        raise ValueError(
            f"A minimum amount of 365 days (= 8760 rows) is required!"
        )
    df_addRows = df[maxAddRowCount - addRowCount : maxAddRowCount]
    df_addRows.index = ["#####" + str(idx) for idx in range(len(df_addRows))]
    df = pd.concat([df, df_addRows])
    return df


def df_to_arr(df: pd.DataFrame) -> tuple[np.ndarray, pd.Index]:
    """
    Converts a Pandas DataFrame to a NumPy array. Preserves the index.
    """
    dfIdx = df.index
    arr = df.to_numpy()
    return arr, dfIdx


def reshape_arr(arr: np.ndarray, dayCount: int) -> np.ndarray:
    """
    Reshapes the input array. The resulting array can be viewed as a collection of heatmaps
    and serves as the input of the GAN. Its shape is (number of profiles, 1, number of hours
    of a day, number of days).
    """
    arr = np.stack([col.reshape(dayCount, -1, 1) for col in arr.T], axis = 3).T
    # arr = np.stack([col.reshape(int(dayCount/4), -1, 1) for col in arr.T], axis = 3).T #alternative aspect ratio
    return arr


def revert_reshape_arr(arr: np.ndarray) -> np.ndarray:
    """
    Reverts the operation of `reshape_arr`.
    """
    arr = arr.T.reshape(-1, arr.shape[0])
    return arr


def min_max_scaler(
    arr: np.ndarray, featureRange: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales the values of the input array between a minimum and a maximum given by
    `featureRange`.

    Args:
        arr (np.ndarray): The input array.
        featureRange (tuple[int, int]): The range used for scaling the data.

    Returns:
        tuple[np.ndarray, np.ndarray]: The resulting array as well as an array containing
        the minimum and the maximum value.
    """
    valMin, valMax = np.min(arr), np.max(arr)
    arr_minMax = np.array([valMin, valMax])
    arr_scaled = (arr - valMin) / (valMax - valMin) * (featureRange[1] - featureRange[0]) + featureRange[0]
    return arr_scaled, arr_minMax


def invert_min_max_scaler(
    arr_scaled: np.ndarray, arr_minMax: np.ndarray, featureRange: tuple[int, int]
) -> np.ndarray:
    """
    Reverts the operation of `min_max_scaler`.
    """
    valMin, valMax = arr_minMax[0], arr_minMax[1]
    arr = (arr_scaled - featureRange[0]) * (valMax - valMin) / (featureRange[1] - featureRange[0]) + valMin
    return arr


def data_prep_wrapper(
    df: pd.DataFrame, dayCount: int, featureRange: tuple[int, int]
) -> tuple:
    """
    A wrapper calling the following functions in the specified order:
    * `add_filler_rows`
    * `df_to_arr`
    * `reshape_arr`
    * `min_max_scaler`

    Args:
        df (pd.DataFrame): A dataframe containing multiple consumption profiles. Each column
            should correspond to one profile and contain hourly values for one year.
        
        dayCount (int): The maximum number of days to be processed by the GAN. If this value
            is changed, the layer structure of the Generator and the Discriminator need to
            be adapted accordingly.
        
        featureRange (tuple[int, int]): The range used for scaling the data.

    Returns:
        tuple[np.ndarray, pd.Index, np.ndarray]: Resulting array, index of the dataframe,
            array containing the minimum and the maximum value.
    """
    df = add_filler_rows(df, dayCount)
    arr, dfIdx = df_to_arr(df)
    arr = reshape_arr(arr, dayCount)
    arr, arr_minMax = min_max_scaler(arr, featureRange)
    return arr, dfIdx, arr_minMax


def get_sep(path: pathlib.Path) -> str:
    """
    Determines and returns the separator used in a CSV file.
    """
    with open(path, newline = "") as file:
        sep = csv.Sniffer().sniff(file.read()).delimiter
        return sep


############################################################################################
############################# Optional (for removing outliers) #############################
############################################################################################


def limit_load_sums(series, alpha):
    colsToRemove = set(series[(series < np.quantile(series, alpha / 2)) | (series > np.quantile(series, 1 - alpha / 2))].index)
    return colsToRemove


def find_outliers(series):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    IQR = q3 - q1
    colsToRemove = set(series[(series < q1 - 1.5 * IQR) | (series > q3 + 1.5 * IQR)].index)
    return colsToRemove


def outlier_removal_wrapper(df, alpha):
    loadSums = df.sum()
    initialColCount = df.shape[1]
    colsToRemove = limit_load_sums(loadSums, alpha) | find_outliers(df.max())
    df = df.drop(columns=colsToRemove)
    print(f"Outlier detection: {len(colsToRemove)} profiles were removed ({initialColCount} → {initialColCount - len(colsToRemove)}).")
    return df