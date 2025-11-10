import json
import pathlib
import pickle
import time
from typing import Dict, List, Optional, Sequence, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

from data_io import (
    LABEL_DICT,
    create_1000_labels_df,
    create_labels_for_100_000_labels_df,
    create_labels_for_50_000_labels_df,
    create_labels_for_5000_synthetic_profiles_per_type,
    create_labels_for_original_shape_synthetic_profiles,
    load_10_000_unlabeled_synthetic_profiles,
    load_synthetic_1000_profiles_per_type,
    load_100_000_synthetic_profiles_per_type,
    load_1300_unlabeled_synthetic_profiles,
    load_5000_synthetic_profiles_per_type,
    load_50_000_synthetic_profiles_per_type,
    load_synthetic_profiles_in_originial_shape,
)
from features import calculate_features_cached


BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "identification_models"
DEFAULT_MODEL_NAME = BASE_DIR / "xgboost_classifier"


def ensure_model_directory() -> None:
    """Ensure the model directory exists."""
    if not MODEL_DIR.exists():
        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Created model directory: {MODEL_DIR}")


def save_model(
    model: XGBClassifier,
    label_encoder: LabelEncoder,
    classification_report_dict: Dict[str, Dict[str, float]],
    model_name: pathlib.Path = DEFAULT_MODEL_NAME,
    cv_fold_metrics: Optional[List[Dict[str, float]]] = None,
) -> None:
    """Save the trained model, label encoder, and classification report."""
    ensure_model_directory()

    model_path = MODEL_DIR / f"{model_name}_model.joblib"
    encoder_path = MODEL_DIR / f"{model_name}_label_encoder.joblib"
    report_path = MODEL_DIR / f"{model_name}_classification_report.pkl"
    cv_metrics_path = MODEL_DIR / f"{model_name}_cv_metrics.pkl"

    joblib.dump(model, model_path)
    joblib.dump(label_encoder, encoder_path)

    with open(report_path, "wb") as f:
        pickle.dump(classification_report_dict, f)

    if cv_fold_metrics is not None:
        with open(cv_metrics_path, "wb") as f:
            pickle.dump(cv_fold_metrics, f)
        print(f"   CV Metrics: {cv_metrics_path}")

    print("✅ Model saved successfully:")
    print(f"   Model: {model_path}")
    print(f"   Label Encoder: {encoder_path}")
    print(f"   Classification Report: {report_path}")

def model_exists(model_name: pathlib.Path = DEFAULT_MODEL_NAME) -> bool:
    """Check if a saved model exists."""
    model_path = MODEL_DIR / f"{model_name}_model.joblib"
    encoder_path = MODEL_DIR / f"{model_name}_label_encoder.joblib"
    report_path = MODEL_DIR / f"{model_name}_classification_report.pkl"
    report_path_json = MODEL_DIR / f"{model_name}_classification_report.json"
    return model_path.exists() and encoder_path.exists() and (report_path.exists() or report_path_json.exists())


def load_model(
    model_name: pathlib.Path = DEFAULT_MODEL_NAME,
) -> Tuple[XGBClassifier, LabelEncoder, Dict[str, Dict[str, float]], Optional[List[Dict[str, float]]]]:
    """Load a saved model, label encoder, and classification report."""
    if not model_exists(model_name):
        raise FileNotFoundError(f"Model '{model_name}' not found in {MODEL_DIR}")

    model_path = MODEL_DIR / f"{model_name}_model.joblib"
    encoder_path = MODEL_DIR / f"{model_name}_label_encoder.joblib"
    report_path = MODEL_DIR / f"{model_name}_classification_report.pkl"
    cv_metrics_path = MODEL_DIR / f"{model_name}_cv_metrics.pkl"

    model = joblib.load(model_path)
    label_encoder = joblib.load(encoder_path)

    report_path_json = MODEL_DIR / f"{model_name}_classification_report.json"
    if report_path.exists():
        with open(report_path, "rb") as f:
            classification_report_dict = pickle.load(f)
    elif report_path_json.exists():
        with open(report_path_json, "r") as f:
            classification_report_dict = json.load(f)
        with open(report_path, "wb") as f:
            pickle.dump(classification_report_dict, f)
        print(f"🔄 Migrated {model_name} classification report from JSON to pickle format")
    else:
        raise FileNotFoundError(f"Classification report file not found for {model_name}")

    cv_fold_metrics = None
    if cv_metrics_path.exists():
        with open(cv_metrics_path, "rb") as f:
            cv_fold_metrics = pickle.load(f)
    else:
        print(f"⚠️ CV metrics file not found for {model_name}.")

    print("✅ Model loaded successfully:")
    print(f"   Model: {model_path}")
    print(f"   Label Encoder: {encoder_path}")
    print(f"   Classification Report: {report_path}")

    return model, label_encoder, classification_report_dict, cv_fold_metrics


def get_or_train_real_data_model(
    X_real: pd.DataFrame,
    y_real: pd.Series,
    model_name: str = "real_data_classifier",
    force_retrain: bool = False,
):
    """Get a trained model for real data or train a new one."""
    if not force_retrain and model_exists(model_name):
        print(f"🔄 Loading existing model '{model_name}'...")
        model, label_encoder, classification_report_dict, cv_fold_metrics = load_model(model_name)

        X_train_subset, X_test, y_train_subset, y_test = train_test_split(
            X_real, y_real, test_size=0.2, random_state=42, stratify=y_real
        )
        if not cv_fold_metrics:
            cv_fold_metrics = collect_cv_fold_metrics(
                X_train_subset,
                y_train_subset,
                n_splits=5,
                random_seeds=(42,),
            )
            save_model(model, label_encoder, classification_report_dict, model_name, cv_fold_metrics=cv_fold_metrics)

        return model, X_test, y_test, label_encoder, classification_report_dict, cv_fold_metrics

    if force_retrain:
        print(f"🔄 Force retraining model '{model_name}'...")
    else:
        print(f"🔄 Training new model '{model_name}'...")

    model, X_test, y_test, label_encoder, classification_report_dict, cv_fold_metrics = create_real_data_classifier(
        X_real, y_real
    )
    save_model(model, label_encoder, classification_report_dict, model_name, cv_fold_metrics=cv_fold_metrics)
    return model, X_test, y_test, label_encoder, classification_report_dict, cv_fold_metrics


def list_saved_models() -> List[str]:
    """List all saved models in the model directory."""
    if not MODEL_DIR.exists():
        print(f"Model directory '{MODEL_DIR}' does not exist.")
        return []

    model_files = [f for f in MODEL_DIR.iterdir() if f.name.endswith("_model.joblib")]
    model_names = [f.name.replace("_model.joblib", "") for f in model_files]

    if model_names:
        print(f"Available saved models in '{MODEL_DIR}':")
        for name in model_names:
            print(f"  - {name}")
    else:
        print(f"No saved models found in '{MODEL_DIR}'.")

    return model_names


def train_XGBoost_optimized(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_state: int = 42,
    return_best_model: bool = True,
):
    """Optimized XGBoost training with K-Fold cross-validation."""
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    print(f"Number of classes: {len(np.unique(y_encoded))}")
    print(f"Encoded class distribution: {pd.Series(y_encoded).value_counts().to_dict()}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    best_accuracy = float("-inf")
    best_xgb_model = None
    fold_metrics = []

    xgb_params = {
        "objective": "multi:softmax",
        "num_class": len(np.unique(y_encoded)),
        "eval_metric": "mlogloss",
        "random_state": random_state,
        "n_jobs": -1,
        "verbosity": 0,
    }

    for fold, (train_index, test_index) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y_encoded[train_index], y_encoded[test_index]

        unique_train_classes = len(np.unique(y_train))
        if unique_train_classes < 2:
            print(f"Warning: Fold {fold + 1} has only {unique_train_classes} class(es)")
            continue

        xgb_model = XGBClassifier(**xgb_params)
        xgb_model.fit(X_train, y_train)

        y_pred = xgb_model.predict(X_test)
        fold_accuracy = accuracy_score(y_test, y_pred)

        macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        macro_precision = precision_score(y_test, y_pred, average="macro", zero_division=0)
        macro_recall = recall_score(y_test, y_pred, average="macro", zero_division=0)

        fold_metrics.append(
            {
                "fold": fold + 1,
                "random_state": random_state,
                "macro_f1": macro_f1,
                "macro_precision": macro_precision,
                "macro_recall": macro_recall,
                "accuracy": fold_accuracy,
            }
        )

        if fold_accuracy > best_accuracy:
            best_accuracy = fold_accuracy
        if return_best_model and fold_accuracy == best_accuracy:
            best_accuracy = fold_accuracy
            best_xgb_model = xgb_model

        print(f"Fold {fold + 1} Accuracy: {fold_accuracy:.4f}")

    print(f"Best Fold Accuracy: {best_accuracy:.4f}")
    if return_best_model:
        return best_xgb_model, fold_metrics
    return None, fold_metrics


def train_XGBoost_with_proper_split_optimized(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    collect_cv_metrics: bool = False,
    cv_n_splits: int = 5,
    cv_random_seeds: Sequence[int] = (42,),
):
    """Optimized version of training with a proper train/test split."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    print(f"Training set size: {len(X_train)} ({(1 - test_size) * 100:.0f}%)")
    print(f"Test set size: {len(X_test)} ({test_size * 100:.0f}%)")

    label_encoder = LabelEncoder()
    label_encoder.fit(y_train)

    best_xgb_model, initial_fold_metrics = train_XGBoost_optimized(
        X_train,
        y_train,
        n_splits=cv_n_splits,
        random_state=random_state,
        return_best_model=True,
    )

    collected_fold_metrics = None
    if collect_cv_metrics:
        collected_fold_metrics = collect_cv_fold_metrics(
            X_train,
            y_train,
            n_splits=cv_n_splits,
            random_seeds=cv_random_seeds,
            reuse_metrics=initial_fold_metrics,
            reused_seed=random_state,
        )

    print("Training completed successfully!")
    return best_xgb_model, X_test, y_test, label_encoder, collected_fold_metrics


def collect_cv_fold_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    n_splits: int = 5,
    random_seeds: Sequence[int] = (42,),
    reuse_metrics: Optional[List[Dict[str, float]]] = None,
    reused_seed: Optional[int] = None,
) -> List[Dict[str, float]]:
    """Run cross-validation for multiple random seeds and aggregate metrics."""
    aggregated_metrics: List[Dict[str, float]] = []
    if reuse_metrics:
        aggregated_metrics.extend(reuse_metrics)

    seen_seed = reused_seed if reuse_metrics else None

    for seed in random_seeds:
        if seen_seed is not None and seed == seen_seed:
            continue
        _, fold_metrics = train_XGBoost_optimized(
            X,
            y,
            n_splits=n_splits,
            random_state=seed,
            return_best_model=False,
        )
        aggregated_metrics.extend(fold_metrics)

    return aggregated_metrics


def create_real_data_classifier(
    X_real: pd.DataFrame,
    y_real: pd.Series,
):
    """Create a classifier for real data."""
    print("🎯 Training XGBoost on real data and evaluating performance...")
    best_xgb_model, X_test, y_test, label_encoder, cv_fold_metrics = train_XGBoost_with_proper_split_optimized(
        X_real,
        y_real,
        test_size=0.2,
        random_state=42,
        collect_cv_metrics=True,
        cv_n_splits=5,
        cv_random_seeds=(42,),
    )

    y_pred_encoded = best_xgb_model.predict(X_test)
    y_pred = label_encoder.inverse_transform(y_pred_encoded)
    classification_report_dict = classification_report(y_test, y_pred, output_dict=True)

    return best_xgb_model, X_test, y_test, label_encoder, classification_report_dict, cv_fold_metrics


def calculate_all_synthetic_features(
    scenario_name: str,
    force_recalculate_features: bool = False,
) -> List[Tuple[str, pd.DataFrame, pd.Series, str]]:
    labels_df_1300 = create_labels_for_original_shape_synthetic_profiles()
    synthetic_time_series_1300 = load_synthetic_profiles_in_originial_shape()
    X_synthetic_1300, y_synthetic_1300 = calculate_features_cached(
        labels_df_1300,
        synthetic_time_series_1300,
        scenario_name,
        cache_key=f"synthetic_1300_{scenario_name}",
        force_recalculate=force_recalculate_features,
    )

    labels_df_5000 = create_1000_labels_df()
    synthetic_time_series_5000 = load_synthetic_1000_profiles_per_type()
    X_synthetic_5000, y_synthetic_5000 = calculate_features_cached(
        labels_df_5000,
        synthetic_time_series_5000,
        scenario_name,
        cache_key=f"synthetic_5000_{scenario_name}",
        force_recalculate=force_recalculate_features,
    )

    labels_df_25000 = create_labels_for_5000_synthetic_profiles_per_type()
    synthetic_time_series_25000 = load_5000_synthetic_profiles_per_type()
    X_synthetic_25000, y_synthetic_25000 = calculate_features_cached(
        labels_df_25000,
        synthetic_time_series_25000,
        scenario_name,
        cache_key=f"synthetic_25000_{scenario_name}",
        force_recalculate=force_recalculate_features,
    )

    labels_df_50000 = create_labels_for_50_000_labels_df()
    synthetic_time_series_50000 = load_50_000_synthetic_profiles_per_type()
    X_synthetic_50000, y_synthetic_50000 = calculate_features_cached(
        labels_df_50000,
        synthetic_time_series_50000,
        scenario_name,
        cache_key=f"synthetic_50000_{scenario_name}",
        force_recalculate=force_recalculate_features,
    )

    labels_df_100000 = create_labels_for_100_000_labels_df()
    synthetic_time_series_100000 = load_100_000_synthetic_profiles_per_type()
    X_synthetic_100000, y_synthetic_100000 = calculate_features_cached(
        labels_df_100000,
        synthetic_time_series_100000,
        scenario_name,
        cache_key=f"synthetic_100000_{scenario_name}",
        force_recalculate=force_recalculate_features,
    )

    print("✅ All synthetic features pre-calculated and cached!")

    return [
        ("1300", X_synthetic_1300, y_synthetic_1300, "Original Shape Profiles (1,300)"),
        ("4000", X_synthetic_5000, y_synthetic_5000, "1000 Profiles Per Type"),
        ("25000", X_synthetic_25000, y_synthetic_25000, "5000 Profiles Per Type"),
        ("250000", X_synthetic_50000, y_synthetic_50000, "50,000 Profiles Per Type"),
        ("400000", X_synthetic_100000, y_synthetic_100000, "100,000 Profiles Per Type"),
    ]


def train_synthetic_data_models(
    X_real: pd.DataFrame,
    y_real: pd.Series,
    scenario_name: str,
    force_retrain: bool = False,
    force_recalculate_features: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Train models on synthetic datasets of varying sizes and evaluate on real data."""
    print("🚀 OPTIMIZED PERFORMANCE TEST 1")
    print("=" * 50)
    print("📊 Pre-calculating all features with caching...")

    performance_results: Dict[str, Dict[str, float]] = {}
    print("🎯 Using optimized 80/20 train/test split with evaluation on held-out test set")

    experiments = calculate_all_synthetic_features(
        scenario_name, force_recalculate_features=force_recalculate_features
    )

    for key, X_data, y_data, description in experiments:
        print(f"\n{'=' * 50}")
        print(f"Training on {description}...")
        print("=" * 50)

        start_time = time.time()

        model_name = f"synthetic_{key}_classifier_{scenario_name}"

        if not force_retrain and model_exists(model_name):
            print(f"🔄 Loading existing model '{model_name}'...")
            best_model, label_encoder, classification_report_dict, cv_fold_metrics = load_model(model_name)
            if not cv_fold_metrics:
                X_train_subset, _, y_train_subset, _ = train_test_split(
                    X_data,
                    y_data,
                    test_size=0.2,
                    random_state=42,
                    stratify=y_data,
                )
                cv_fold_metrics = collect_cv_fold_metrics(
                    X_train_subset,
                    y_train_subset,
                    n_splits=5,
                    random_seeds=(42,),
                )
                save_model(best_model, label_encoder, classification_report_dict, model_name, cv_fold_metrics=cv_fold_metrics)

        else:
            best_model, X_test, y_test, label_encoder, cv_fold_metrics = train_XGBoost_with_proper_split_optimized(
                X_data,
                y_data,
                test_size=0.2,
                random_state=42,
                collect_cv_metrics=True,
                cv_n_splits=5,
                cv_random_seeds=(42,),
            )
            y_real_encoded = label_encoder.transform(y_real)
            y_real_pred_encoded = best_model.predict(X_real)
            y_real_pred = label_encoder.inverse_transform(y_real_pred_encoded)

            eval_accuracy = accuracy_score(y_real_encoded, y_real_pred_encoded)
            print(f"Evaluation accuracy on real data: {eval_accuracy:.4f}")
            classification_report_dict = classification_report(y_real, y_real_pred, output_dict=True)
            save_model(best_model, label_encoder, classification_report_dict, model_name, cv_fold_metrics=cv_fold_metrics)

        eval_data_source = "Real Data"
        training_time = time.time() - start_time

        classification_metrics = {
            "macro_f1": classification_report_dict["macro avg"]["f1-score"],
            "macro_precision": classification_report_dict["macro avg"]["precision"],
            "macro_recall": classification_report_dict["macro avg"]["recall"],
            "per_class_f1": {
                label: classification_report_dict[label]["f1-score"]
                for label in classification_report_dict
                if label not in ["accuracy", "macro avg", "weighted avg"]
            },
            "per_class_precision": {
                label: classification_report_dict[label]["precision"]
                for label in classification_report_dict
                if label not in ["accuracy", "macro avg", "weighted avg"]
            },
            "per_class_recall": {
                label: classification_report_dict[label]["recall"]
                for label in classification_report_dict
                if label not in ["accuracy", "macro avg", "weighted avg"]
            },
            "data_size": len(y_data),
            "train_size": round(len(y_data) * 0.8),
            "eval_size": round(len(y_real) * 0.2),
            "eval_data_source": eval_data_source,
            "cv_fold_metrics": cv_fold_metrics,
            "description": description,
        }

        performance_results[key] = classification_metrics

        print(f"✅ {description} completed in {training_time:.2f}s")

    print("\n✅ loaded synthetic data models successfully!")

    return performance_results


def label_synthetic_data_and_compare_distribution(
    X_real: pd.DataFrame,
    y_real: pd.Series,
    real_labels_df: pd.DataFrame,
    real_time_series: pd.DataFrame,
    save_path: pathlib.Path,
    scenario_name: str,
    force_retrain: bool = False,
) -> None:
    """Train on real data, label synthetic data, and generate comparison plots."""
    print("\n" + "=" * 70)
    print("SECOND STUDY: Domain Transfer Analysis")
    print("Training XGBoost on Real Data → Labeling Synthetic Profiles")
    print("=" * 70)

    model_name = f"real_data_full_classifier_{scenario_name}"

    if not force_retrain and model_exists(model_name):
        print(f"🔄 Loading existing model '{model_name}'...")
        xgb_real_model, label_encoder_real_full, _, _ = load_model(model_name)
    else:
        print("\n🎯 Step 1: Training XGBoost on ALL real data...")
        print("Real data training set:")
        print(f"  Total samples: {len(y_real)}")
        print(f"  Class distribution: {pd.Series(y_real).value_counts().to_dict()}")

        label_encoder_real_full = LabelEncoder()
        y_real_encoded_full = label_encoder_real_full.fit_transform(y_real)

        xgb_real_model = XGBClassifier(
            objective="multi:softmax",
            num_class=len(np.unique(y_real_encoded_full)),
            eval_metric="mlogloss",
            random_state=42,
        )
        xgb_real_model.fit(X_real, y_real_encoded_full)

        y_real_pred_encoded = xgb_real_model.predict(X_real)
        y_real_pred = label_encoder_real_full.inverse_transform(y_real_pred_encoded)
        classification_report_dict = classification_report(y_real, y_real_pred, output_dict=True)

        save_model(xgb_real_model, label_encoder_real_full, classification_report_dict, model_name)
        print(f"✅ Model trained on real data saved as '{model_name}'")

    print("\n🎯 Step 2: Loading large synthetic dataset and predicting labels...")

    synthetic_time_series_large = load_1300_unlabeled_synthetic_profiles()
    dummy_labels_df = pd.DataFrame(
        {
            "ID": range(1, synthetic_time_series_large.shape[1] + 1),
            "Category": ["NONE"] * synthetic_time_series_large.shape[1],
        }
    )
    X_synthetic_large, _ = calculate_features_cached(
        dummy_labels_df,
        synthetic_time_series_large,
        scenario_name,
        cache_key=f"synthetic_large_unlabeled_{scenario_name}",
        force_recalculate=force_retrain,
    )

    y_synthetic_pred_encoded = xgb_real_model.predict(X_synthetic_large)
    y_synthetic_pred = label_encoder_real_full.inverse_transform(y_synthetic_pred_encoded)

    print(f"Predicted label distribution: {pd.Series(y_synthetic_pred).value_counts().to_dict()}")

    labeled_synthetic_data = load_synthetic_profiles_in_originial_shape()
    synthetic_labels = create_labels_for_original_shape_synthetic_profiles()
    X_synthetic_labeled, _ = calculate_features_cached(
        synthetic_labels,
        labeled_synthetic_data,
        scenario_name,
        cache_key=f"synthetic_labeled_{scenario_name}",
        force_recalculate=force_retrain,
    )
    y_synthetic_labeled_pred_encoded = xgb_real_model.predict(X_synthetic_labeled)
    y_synthetic_labeled_pred = label_encoder_real_full.inverse_transform(y_synthetic_labeled_pred_encoded)
    print(f"Predicted label distribution: {pd.Series(y_synthetic_labeled_pred).value_counts().to_dict()}")

    dummy_labels_df_10000 = pd.DataFrame(
        {
            "ID": range(1, 10_000 + 1),
            "Category": ["NONE"] * 10_000,
        }
    )
    synthetic_data_10000 = load_10_000_unlabeled_synthetic_profiles()
    X_synthetic_10000, _ = calculate_features_cached(
        dummy_labels_df_10000,
        synthetic_data_10000,
        scenario_name,
        cache_key=f"synthetic_unlabeled_10000_{scenario_name}",
        force_recalculate=force_retrain,
    )
    y_synthetic_10000_pred_encoded = xgb_real_model.predict(X_synthetic_10000)
    y_synthetic_10000_pred = label_encoder_real_full.inverse_transform(y_synthetic_10000_pred_encoded)
    print(f"Predicted label distribution: {pd.Series(y_synthetic_10000_pred).value_counts().to_dict()}")

    print("\n🎯 Step 3: Creating label distribution comparison...")

    from paper_figures import (
        create_hourly_consumption_comparison,
        create_hourly_load_boxplot_comparison,
        create_label_distribution_comparison,
    )

    create_label_distribution_comparison(
        real_labels_df["Category"],
        y_synthetic_pred,
        save_path=save_path / "label_distribution_comparison.svg",
    )
    print(f"\n✅ Label distribution comparison saved to: {save_path}")
    create_label_distribution_comparison(
        real_labels_df["Category"],
        y_synthetic_labeled_pred,
        save_path=save_path / "label_distribution_comparison_labeled_1300.svg",
    )
    print(
        "\n✅ Label distribution comparison of labeled synthetic data saved to: figures/label_distribution_comparison_labeled_1300.svg"
    )
    create_label_distribution_comparison(
        real_labels_df["Category"],
        y_synthetic_10000_pred,
        save_path=save_path / "label_distribution_comparison_unlabeled_10000.svg",
    )
    print(
        "\n✅ Label distribution comparison of unlabeled synthetic data saved to: figures/label_distribution_comparison_unlabeled_10000.svg"
    )

    print("\n🎯 Step 4: Creating hourly consumption pattern comparison...")

    create_hourly_consumption_comparison(
        real_labels_df,
        real_time_series,
        y_synthetic_pred,
        synthetic_time_series_large,
        save_path=save_path / "hourly_consumption_comparison.svg",
    )

    create_hourly_consumption_comparison(
        real_labels_df,
        real_time_series,
        y_synthetic_labeled_pred,
        labeled_synthetic_data,
        save_path=save_path / "hourly_consumption_comparison_labeled.svg",
    )

    create_hourly_consumption_comparison(
        real_labels_df,
        real_time_series,
        y_synthetic_10000_pred,
        synthetic_data_10000,
        save_path=save_path / "hourly_consumption_comparison_unlabeled_10_000.svg",
    )
    create_hourly_load_boxplot_comparison(
        real_time_series,
        synthetic_data_10000,
        real_labels_df,
        y_synthetic_10000_pred,
        save_path=save_path / "hourly_consumption_comparison_unlabeled_10_000_boxplot.png",
    )
