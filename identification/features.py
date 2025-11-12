import json
import pathlib
import pickle
import time
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd
from scipy.stats import skew


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
FEATURE_CACHE_DIR = BASE_DIR / "identification_feature_cache"


def ensure_feature_cache_directory() -> None:
    """Ensure the feature cache directory exists."""
    if not FEATURE_CACHE_DIR.exists():
        FEATURE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created feature cache directory: {FEATURE_CACHE_DIR}")


def get_feature_cache_path(cache_key: str) -> pathlib.Path:
    """Get the file path for cached features."""
    return FEATURE_CACHE_DIR / f"{cache_key}_features.pkl"


def get_feature_metadata_path(cache_key: str) -> pathlib.Path:
    """Get the file path for feature metadata."""
    return FEATURE_CACHE_DIR / f"{cache_key}_metadata.pkl"


def feature_cache_exists(cache_key: Optional[str]) -> bool:
    """Check if cached features exist on disk."""
    if not cache_key:
        return False
    feature_path = get_feature_cache_path(cache_key)
    metadata_path = get_feature_metadata_path(cache_key)
    metadata_path_json = FEATURE_CACHE_DIR / f"{cache_key}_metadata.json"
    feature_path_joblib = FEATURE_CACHE_DIR / f"{cache_key}_features.joblib"
    return (feature_path.exists() or feature_path_joblib.exists()) and (
        metadata_path.exists() or metadata_path_json.exists()
    )


def save_features_to_disk(
    X: pd.DataFrame,
    y: pd.Series,
    cache_key: Optional[str],
    labels_df_shape: List[int],
    time_series_shape: List[int],
) -> None:
    """Save features to disk with metadata."""
    if not cache_key:
        return

    ensure_feature_cache_directory()

    feature_path = get_feature_cache_path(cache_key)
    metadata_path = get_feature_metadata_path(cache_key)

    with open(feature_path, "wb") as f:
        pickle.dump((X, y), f)

    metadata = {
        "cache_key": cache_key,
        "labels_df_shape": labels_df_shape,
        "time_series_shape": time_series_shape,
        "feature_shape": X.shape,
        "target_shape": y.shape,
        "timestamp": time.time(),
    }

    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)

    print(f"✅ Features saved to disk: {feature_path}")


def load_features_from_disk(cache_key: str) -> Optional[Tuple[pd.DataFrame, pd.Series, Dict[str, Any]]]:
    """Load features from disk."""
    if not feature_cache_exists(cache_key):
        return None

    feature_path = get_feature_cache_path(cache_key)
    metadata_path = get_feature_metadata_path(cache_key)

    try:
        feature_path_joblib = FEATURE_CACHE_DIR / f"{cache_key}_features.joblib"

        if feature_path.exists():
            with open(feature_path, "rb") as f:
                X, y = pickle.load(f)
        elif feature_path_joblib.exists():
            X, y = joblib.load(feature_path_joblib)
            with open(feature_path, "wb") as f:
                pickle.dump((X, y), f)
            print(f"🔄 Migrated {cache_key} features from joblib to pickle format")
        else:
            raise FileNotFoundError(f"Feature file not found for {cache_key}")

        metadata_path_json = FEATURE_CACHE_DIR / f"{cache_key}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        elif metadata_path_json.exists():
            with open(metadata_path_json, "r") as f:
                metadata = json.load(f)
            with open(metadata_path, "wb") as f:
                pickle.dump(metadata, f)
            print(f"🔄 Migrated {cache_key} metadata from JSON to pickle format")
        else:
            raise FileNotFoundError(f"Metadata file not found for {cache_key}")

        print(f"✅ Features loaded from disk: {feature_path}")
        print(f"   Features shape: {metadata['feature_shape']}")
        print(f"   Cached on: {time.ctime(metadata['timestamp'])}")

        return X, y, metadata
    except Exception as e:
        print(f"⚠️  Error loading cached features for {cache_key}: {e}")
        return None


def validate_cached_features(cache_key: str, labels_df: pd.DataFrame, time_series: pd.DataFrame) -> bool:
    """Validate that cached features match current input data."""
    if not feature_cache_exists(cache_key):
        return False

    try:
        metadata_path = get_feature_metadata_path(cache_key)
        metadata_path_json = FEATURE_CACHE_DIR / f"{cache_key}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        elif metadata_path_json.exists():
            with open(metadata_path_json, "r") as f:
                metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found for {cache_key}")

        current_labels_shape = labels_df.shape
        current_time_series_shape = time_series.shape

        if (
            metadata["labels_df_shape"] != list(current_labels_shape)
            or metadata["time_series_shape"] != list(current_time_series_shape)
        ):
            print(f"⚠️  Cache invalid for {cache_key}: input data shape changed")
            return False

        return True
    except Exception as e:
        print(f"⚠️  Error validating cache for {cache_key}: {e}")
        return False


def calculate_features_optimized(labels_df: pd.DataFrame, time_series: pd.DataFrame, scenario_name: str) -> Tuple[pd.DataFrame, pd.Series]:
    """Optimized feature calculation with vectorized statistics."""
    if not isinstance(time_series.index, pd.DatetimeIndex):
        time_series.index = pd.to_datetime(time_series.index)

    if time_series.isnull().any().any():
        print("Warning: time_series contains NaN values")
        time_series = time_series.fillna(0)

    def extract_time_features_optimized(df: pd.DataFrame, freq: str, prefix: str) -> pd.DataFrame:
        resampled = df.resample(freq)
        mean = resampled.mean()
        std = resampled.std()
        max_ = resampled.max()
        skew_values = resampled.apply(pd.Series.skew)

        with np.errstate(divide="ignore", invalid="ignore"):
            peak_to_avg = max_ / mean
            peak_to_avg = peak_to_avg.replace([np.inf, -np.inf], 0).fillna(0)

        n_periods = len(mean)
        mean.index = [f"{prefix}_mean_{i}" for i in range(n_periods)]
        std.index = [f"{prefix}_std_{i}" for i in range(n_periods)]
        skew_values.index = [f"{prefix}_skew_{i}" for i in range(n_periods)]
        max_.index = [f"{prefix}_max_{i}" for i in range(n_periods)]
        peak_to_avg.index = [f"{prefix}_peak_to_avg_{i}" for i in range(n_periods)]

        sum_ = resampled.sum()
        sum_.index = [f"{prefix}_sum_{i}" for i in range(n_periods)]
        features = pd.concat([mean, std, skew_values, max_, sum_, peak_to_avg], axis=0)

        features = features.reset_index().rename(columns={"index": "feature"})
        return features

    print("Extracting weekly features...")
    weekly_features = extract_time_features_optimized(time_series, "W", "W")

    print("Extracting monthly features...")
    # Pandas expects standard aliases (e.g., 'M' for month end)
    monthly_features = extract_time_features_optimized(time_series, "M", "M")

    print("Extracting annual features...")
    annual_features = extract_time_features_optimized(time_series, "A", "A")

    if "daily" in scenario_name:
        print("================================================")
        print("DAILY FEATURES ARE ADDED")
        print("================================================")
        daily_features = extract_time_features_optimized(time_series, "D", "D")
        combined_features = pd.concat(
            [daily_features, weekly_features, monthly_features, annual_features], axis=0, ignore_index=True
        )
    else:
        combined_features = pd.concat(
            [weekly_features, monthly_features, annual_features], axis=0, ignore_index=True
        )

    feature_names = combined_features["feature"].values
    combined_features_temp = combined_features.drop(columns=["feature"]).transpose()
    combined_features_temp.columns = feature_names
    combined_features_temp = combined_features_temp.reset_index().rename(columns={"index": "ID"})
    combined_features_temp["ID"] = combined_features_temp["ID"].astype(int)

    if combined_features_temp.isnull().any().any():
        print("Warning: combined_features contains NaN values, filling with 0")
        combined_features_temp = combined_features_temp.fillna(0)

    data_synthetic = pd.merge(combined_features_temp, labels_df[["ID", "Category"]], on="ID")
    X = data_synthetic.drop(columns=["Category", "ID"])
    y = data_synthetic["Category"]

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")

    return X, y


def calculate_features_cached(
    labels_df: pd.DataFrame,
    time_series: pd.DataFrame,
    scenario_name: str,
    cache_key: Optional[str] = None,
    force_recalculate: bool = False,
) -> Tuple[pd.DataFrame, pd.Series]:
    """Cached version of feature calculation with disk persistence."""
    if not cache_key:
        print("No cache key provided, calculating features without caching")
        return calculate_features_optimized(labels_df, time_series, scenario_name)

    if not force_recalculate and feature_cache_exists(cache_key):
        if validate_cached_features(cache_key, labels_df, time_series):
            result = load_features_from_disk(cache_key)
            if result is not None:
                X, y, _ = result
                print(f"💾 Loaded features from disk cache for {cache_key}")
                return X, y
        else:
            print(f"🗑️ Cached features invalid for {cache_key}, recalculating...")

    if force_recalculate:
        print(f"🔄 Force recalculating features for {cache_key}")
    else:
        print(f"⚙️ Calculating features for {cache_key or 'unknown dataset'}")

    start_time = time.time()
    X, y = calculate_features_optimized(labels_df, time_series, scenario_name)
    calculation_time = time.time() - start_time

    print(f"✅ Feature calculation completed in {calculation_time:.2f} seconds")

    save_features_to_disk(X, y, cache_key, list(labels_df.shape), list(time_series.shape))
    print(f"🗃️ Features cached to disk for {cache_key}")

    return X, y


def clear_disk_feature_cache() -> None:
    """Clear all cached features from disk."""
    if not FEATURE_CACHE_DIR.exists():
        print("📁 Feature cache directory doesn't exist")
        return

    try:
        import shutil

        shutil.rmtree(FEATURE_CACHE_DIR)
        print(f"🗑️ Disk feature cache cleared: {FEATURE_CACHE_DIR}")
    except Exception as e:
        print(f"⚠️ Error clearing disk cache: {e}")


def list_cached_features() -> List[str]:
    """List all cached features on disk."""
    if not FEATURE_CACHE_DIR.exists():
        print(f"📁 Feature cache directory '{FEATURE_CACHE_DIR}' does not exist")
        return []

    cache_files_pkl = [f for f in FEATURE_CACHE_DIR.iterdir() if f.name.endswith("_features.pkl")]
    cache_files_joblib = [f for f in FEATURE_CACHE_DIR.iterdir() if f.name.endswith("_features.joblib")]

    cache_keys_pkl = [f.stem.replace("_features", "") for f in cache_files_pkl]
    cache_keys_joblib = [f.stem.replace("_features", "") for f in cache_files_joblib]

    cache_keys = list(set(cache_keys_pkl + cache_keys_joblib))

    if cache_keys:
        print(f"📋 Available cached features in '{FEATURE_CACHE_DIR}':")
        for key in cache_keys:
            try:
                metadata_path = get_feature_metadata_path(key)
                metadata_path_json = FEATURE_CACHE_DIR / f"{key}_metadata.json"
                feature_path_pkl = get_feature_cache_path(key)
                feature_path_joblib = FEATURE_CACHE_DIR / f"{key}_features.joblib"

                if feature_path_pkl.exists():
                    format_type = "pickle"
                elif feature_path_joblib.exists():
                    format_type = "joblib (legacy)"
                else:
                    format_type = "unknown"

                if metadata_path.exists():
                    with open(metadata_path, "rb") as f:
                        metadata = pickle.load(f)
                    print(
                        f"  - {key}: {metadata['feature_shape']} features, cached {time.ctime(metadata['timestamp'])} ({format_type})"
                    )
                elif metadata_path_json.exists():
                    with open(metadata_path_json, "r") as f:
                        metadata = json.load(f)
                    print(
                        f"  - {key}: {metadata['feature_shape']} features, cached {time.ctime(metadata['timestamp'])} ({format_type})"
                    )
                else:
                    print(f"  - {key}: (metadata missing, {format_type})")
            except Exception as e:
                print(f"  - {key}: (error reading metadata: {e})")
    else:
        print(f"📁 No cached features found in '{FEATURE_CACHE_DIR}'")

    return cache_keys


def get_cache_info(cache_key: str) -> Optional[Dict[str, Any]]:
    """Get detailed information about a cached feature set."""
    if not feature_cache_exists(cache_key):
        print(f"❌ Cache '{cache_key}' does not exist")
        return None

    try:
        metadata_path = get_feature_metadata_path(cache_key)
        metadata_path_json = FEATURE_CACHE_DIR / f"{cache_key}_metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        elif metadata_path_json.exists():
            with open(metadata_path_json, "r") as f:
                metadata = json.load(f)
        else:
            raise FileNotFoundError(f"Metadata file not found for {cache_key}")

        feature_path = get_feature_cache_path(cache_key)
        file_size = feature_path.stat().st_size / (1024 * 1024)

        print(f"📊 Cache info for '{cache_key}':")
        print(f"   Features shape: {metadata['feature_shape']}")
        print(f"   Labels shape: {metadata['target_shape']}")
        print(f"   Input data shape: {metadata['labels_df_shape']} x {metadata['time_series_shape']}")
        print(f"   File size: {file_size:.2f} MB")
        print(f"   Cached on: {time.ctime(metadata['timestamp'])}")

        return metadata
    except Exception as e:
        print(f"⚠️ Error reading cache info for {cache_key}: {e}")
        return None


def calc_features(arr: np.ndarray, axis: int) -> np.ndarray:
    """Calculate statistical features of the input array safely."""
    if np.isnan(arr).any():
        print("Warning: NaN values detected in input array for feature calculation")
        arr = np.nan_to_num(arr, nan=0.0)

    arr_features = np.zeros((9, arr.shape[1] if axis == 0 else arr.shape[0]), dtype=np.float32)

    try:
        arr_features[0] = np.mean(arr, axis=axis)
        std = np.std(arr, axis=axis)
        arr_features[1] = np.nan_to_num(std, nan=0.0)
        arr_features[2] = np.min(arr, axis=axis)
        arr_features[3] = np.max(arr, axis=axis)
        arr_features[4] = np.median(arr, axis=axis)

        try:
            skewness = skew(arr, axis=axis)
            arr_features[5] = np.nan_to_num(skewness, nan=0.0, posinf=0.0, neginf=0.0)
        except Exception as err:
            print(f"Warning: Error calculating skewness: {err}")
            arr_features[5] = 0.0

        arr_features[6] = np.ptp(arr, axis=axis)

        try:
            arr_features[7] = np.percentile(arr, 25, axis=axis)
            arr_features[8] = np.percentile(arr, 75, axis=axis)
        except Exception as err:
            print(f"Warning: Error calculating quartiles: {err}")
            arr_features[7] = arr_features[2]
            arr_features[8] = arr_features[3]

    except Exception as err:
        print(f"Warning: Error in feature calculation: {err}")
        return np.zeros((9, arr.shape[1] if axis == 0 else arr.shape[0]), dtype=np.float32)

    if np.isnan(arr_features).any():
        print("Warning: NaN values detected in calculated features")
        arr_features = np.nan_to_num(arr_features, nan=0.0)

    return arr_features
