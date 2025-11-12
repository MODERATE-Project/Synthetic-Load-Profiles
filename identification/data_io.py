import csv
import pathlib
from typing import Dict, List
import numpy as np
import pandas as pd


LABEL_DICT: Dict[str, str] = {
    "PV": "PV",
    "heat_pump+PV": "HP+PV",
    "EV": "EV",
    "EV+PV": "EV+PV",
    "None": "None",
}


def get_sep(path: pathlib.Path) -> str:
    """Determine the delimiter used in a CSV file."""
    with open(path, newline="") as file:
        sep = csv.Sniffer().sniff(file.read()).delimiter
        return sep


def load_10_000_unlabeled_synthetic_profiles(path: pathlib.Path) -> pd.DataFrame:
    """Load the 10,000 unlabeled synthetic profiles."""
    return pd.read_parquet(path / "10000_profiles_all.parquet")


def load_synthetic_1000_profiles_per_type(path: pathlib.Path) -> pd.DataFrame:
    """Load the synthetic 1,000 profiles per type (5,000 total)."""
    df_list: List[pd.DataFrame] = []
    for label_key in LABEL_DICT:
        df = pd.read_parquet(path / f"1000_profiles_{label_key}.parquet")
        df_list.append(df)

    synthetic_time_series = pd.concat(df_list, axis=1)
    synthetic_time_series.columns = np.arange(1, 5_001)
    synthetic_time_series.index = pd.to_datetime(synthetic_time_series.index)
    return synthetic_time_series


def load_5000_synthetic_profiles_per_type(path: pathlib.Path) -> pd.DataFrame:
    """Load the synthetic 5,000 profiles per type (25,000 total)."""
    df_list: List[pd.DataFrame] = []
    for label_key in LABEL_DICT:
        df = pd.read_parquet(path / f"5000_profiles_{label_key}.parquet")
        df_list.append(df)

    synthetic_time_series = pd.concat(df_list, axis=1)
    synthetic_time_series.columns = np.arange(1, synthetic_time_series.shape[1] + 1)
    synthetic_time_series.index = pd.to_datetime(synthetic_time_series.index)
    return synthetic_time_series


def load_50_000_synthetic_profiles_per_type(path: pathlib.Path) -> pd.DataFrame:
    """Load the synthetic 50,000 profiles per type (250,000 total)."""
    df_list: List[pd.DataFrame] = []
    for label_key in LABEL_DICT:
        df = pd.read_parquet(path / f"50000_profiles_{label_key}.parquet")
        df_list.append(df)

    synthetic_time_series = pd.concat(df_list, axis=1)
    synthetic_time_series.columns = np.arange(1, synthetic_time_series.shape[1] + 1)
    synthetic_time_series.index = pd.to_datetime(synthetic_time_series.index)
    return synthetic_time_series


def load_100_000_synthetic_profiles_per_type(path: pathlib.Path) -> pd.DataFrame:
    """Load the synthetic 100,000 profiles per type (500,000 total)."""
    df_list: List[pd.DataFrame] = []
    for label_key in LABEL_DICT:
        df = pd.read_parquet(path / f"100000_profiles_{label_key}.parquet")
        df_list.append(df)

    synthetic_time_series = pd.concat(df_list, axis=1)
    synthetic_time_series.columns = np.arange(1, synthetic_time_series.shape[1] + 1)
    synthetic_time_series.index = pd.to_datetime(synthetic_time_series.index)
    return synthetic_time_series


def load_synthetic_profiles_in_originial_shape(path: pathlib.Path) -> pd.DataFrame:
    """Load the synthetic profiles in their original shape (1,300 profiles)."""
    df_list: List[pd.DataFrame] = []
    for label_key in LABEL_DICT:
        if label_key == "EV":
            df = pd.read_parquet(path / f"100_profiles_{label_key}.parquet")
        else:
            df = pd.read_parquet(path / f"300_profiles_{label_key}.parquet")
        df_list.append(df)

    synthetic_time_series = pd.concat(df_list, axis=1)
    synthetic_time_series.columns = np.arange(1, 1_301)
    synthetic_time_series.index = pd.to_datetime(synthetic_time_series.index)
    return synthetic_time_series


def load_1300_unlabeled_synthetic_profiles(path: pathlib.Path) -> pd.DataFrame:
    """Load the 1,300 unlabeled synthetic profiles."""
    return pd.read_parquet(path / "1300_profiles_all.parquet")


def load_real_data(path: pathlib.Path) -> pd.DataFrame:
    """Load the real Fluvius data."""
    csv_path = path / "fluvius_wide_format.csv"
    df = pd.read_csv(csv_path, sep=get_sep(csv_path), index_col=0)
    df.columns = [i for i in range(1, df.shape[1] + 1)]
    return df


def load_1300_synthetic_profiles(path: pathlib.Path) -> pd.DataFrame:
    """Load the labeled 1,300 synthetic profiles."""
    return pd.read_parquet(path / "1300_profiles_all.parquet")


def load_10_000_synthetic_profiles(path: pathlib.Path) -> pd.DataFrame:
    """Load the labeled 10,000 synthetic profiles."""
    return pd.read_parquet(path / "10000_profiles_all.parquet")


def create_1000_labels_df() -> pd.DataFrame:
    labels = [1000 * [label] for label in LABEL_DICT.values()]
    flat_list = [item for sublist in labels for item in sublist]
    return pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 5_001)})


def create_labels_for_50_000_labels_df() -> pd.DataFrame:
    labels = [50_000 * [label] for label in LABEL_DICT.values()]
    flat_list = [item for sublist in labels for item in sublist]
    return pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 250_001)})


def create_labels_for_100_000_labels_df() -> pd.DataFrame:
    labels = [100_000 * [label] for label in LABEL_DICT.values()]
    flat_list = [item for sublist in labels for item in sublist]
    return pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 500_001)})


def create_labels_for_original_shape_synthetic_profiles() -> pd.DataFrame:
    labels: List[List[str]] = []
    for label_key, label_value in LABEL_DICT.items():
        if label_key == "EV":
            labels.append(100 * [label_value])
        else:
            labels.append(300 * [label_value])

    flat_list = [item for sublist in labels for item in sublist]
    return pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 1_301)})


def create_labels_for_5000_synthetic_profiles_per_type() -> pd.DataFrame:
    labels = [5000 * [label] for label in LABEL_DICT.values()]
    flat_list = [item for sublist in labels for item in sublist]
    return pd.DataFrame({"Category": flat_list, "ID": np.arange(1, 25_001)})
