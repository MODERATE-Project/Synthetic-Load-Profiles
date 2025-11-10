import pathlib
import pandas as pd

from features import calculate_features_cached
from identification_models import (
    get_or_train_real_data_model,
    label_synthetic_data_and_compare_distribution,
    train_synthetic_data_models,
)
from paper_figures import (
    compare_model_with_different_synthetic_training_sizes,
    create_category_feature_statistics_plots,
    create_real_data_performance_figure,
    plot_cv_metric_dispersion,
)
from data_io import (
    create_labels_for_original_shape_synthetic_profiles,
    load_synthetic_profiles_in_originial_shape,
)


PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
FIGURE_DIR = PROJECT_ROOT / "paper_figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

def main(scenario_name: str, force_retrain: bool = False, force_recalculate_features: bool = False) -> None:

    data_dir = PROJECT_ROOT / "data" / "sample_data" / "opendata_fluvius"
    real_labels_df = pd.read_csv(data_dir / "fluvius_indicators.csv")
    real_labels_df.rename(columns={"EAN_ID": "ID", "label": "Category"}, inplace=True)
    real_labels_df["Category"] = real_labels_df["Category"].map(
        {"standard": "None", "PV": "PV", "heat pump+PV": "HP+PV", "EV": "EV", "EV+PV": "EV+PV"}
    )
    real_time_series = pd.read_csv(data_dir / "fluvius_wide_format.csv", index_col=0)

    print("🚀 Pre-calculating features for all datasets (optimization)...")
    print("  Calculating real data features...")
    X_real, y_real = calculate_features_cached(
        real_labels_df,
        real_time_series,
        scenario_name,
        cache_key=f"real_data_{scenario_name}",
        force_recalculate=force_recalculate_features,
    )

    print("  Calculating synthetic dataset features...")

    print("\n" + "=" * 70)
    print("Creating real data performance visualization...")
    print("=" * 70)

    print(f"Force retrain: {force_retrain}")
    print(f"Force recalculate features: {force_recalculate_features}")

    (
        real_model,
        X_test_real,
        y_test_real,
        label_encoder_real,
        classification_report_dict_real_data,
        real_cv_fold_metrics,
    ) = get_or_train_real_data_model(
        X_real,
        y_real,
        model_name=f"real_data_classifier_{scenario_name}",
        force_retrain=force_retrain,
    )

    performance_results_real_data = {
        "macro_f1": classification_report_dict_real_data["macro avg"]["f1-score"],
        "macro_precision": classification_report_dict_real_data["macro avg"]["precision"],
        "macro_recall": classification_report_dict_real_data["macro avg"]["recall"],
        "per_class_f1": {
            label: classification_report_dict_real_data[label]["f1-score"]
            for label in classification_report_dict_real_data.keys()
            if label not in ["accuracy", "macro avg", "weighted avg"]
        },
        "per_class_precision": {
            label: classification_report_dict_real_data[label]["precision"]
            for label in classification_report_dict_real_data.keys()
            if label not in ["accuracy", "macro avg", "weighted avg"]
        },
        "per_class_recall": {
            label: classification_report_dict_real_data[label]["recall"]
            for label in classification_report_dict_real_data.keys()
            if label not in ["accuracy", "macro avg", "weighted avg"]
        },
        "data_size": len(y_real),
        "train_size": len(y_real) - len(y_test_real),
        "eval_size": len(y_real),
        "eval_data_source": "Real Data",
        "cv_fold_metrics": real_cv_fold_metrics,
        "description": "Real Data (80% training folds)",
    }

    performance_results_synthetic_data_models = train_synthetic_data_models(
        X_real,
        y_real,
        scenario_name=scenario_name,
        force_retrain=force_retrain,
        force_recalculate_features=force_recalculate_features,
    )

    print("\n" + "=" * 70)
    print("Creating statistical feature comparisons for each category...")
    print("=" * 70)
    synthetic_time_series_original = load_synthetic_profiles_in_originial_shape()
    synthetic_labels_original = create_labels_for_original_shape_synthetic_profiles()
    create_category_feature_statistics_plots(
        real_time_series,
        real_labels_df,
        synthetic_time_series_original,
        synthetic_labels_original,
        save_dir="figures",
        scenario_tag=f"{scenario_name}_synthetic1300",
    )

    cv_dispersion_inputs = {}
    if performance_results_real_data.get("cv_fold_metrics"):
        cv_dispersion_inputs[performance_results_real_data.get("description", "Real Data")] = performance_results_real_data[
            "cv_fold_metrics"
        ]
    for key, metrics in performance_results_synthetic_data_models.items():
        label = metrics.get("description", key)
        if metrics.get("cv_fold_metrics"):
            cv_dispersion_inputs[label] = metrics["cv_fold_metrics"]

    plot_cv_metric_dispersion(
        cv_dispersion_inputs,
        save_path=FIGURE_DIR / f"cv_metric_dispersion_{scenario_name}.svg",
    )

    compare_model_with_different_synthetic_training_sizes(
        performance_results_synthetic_data_models,
        performance_results_real_data,
        save_path=FIGURE_DIR / f"performance_comparison_{scenario_name}.svg",
    )

    create_real_data_performance_figure(
        classification_report_dict_real_data,
        performance_results_synthetic_data_models,
        save_path=FIGURE_DIR / f"real_data_performance_real_model_{scenario_name}.svg",
    )

    label_synthetic_data_and_compare_distribution(
        X_real,
        y_real,
        real_labels_df,
        real_time_series,
        save_path=FIGURE_DIR / f"label_distribution_comparison_{scenario_name}.svg",
        scenario_name=scenario_name,
        force_retrain=force_retrain,
    )


if __name__ == "__main__":
    main(scenario_name="v2", force_retrain=True, force_recalculate_features=True)
