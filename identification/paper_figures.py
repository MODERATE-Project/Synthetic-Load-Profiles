import pathlib
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix

from features import calc_features


STAT_FEATURE_LABELS = [
    "Mean",
    "Standard Deviation",
    "Minimum",
    "Maximum",
    "Median",
    "Skewness",
    "Peak-to-Peak",
    "25th Percentile",
    "75th Percentile",
]


def get_consistent_color_mapping() -> Dict[str, str]:
    """Return a consistent color mapping for all class categories."""
    return {
        "EV": "#1f77b4",
        "EV+PV": "#ff7f0e",
        "PV": "#2ca02c",
        "PV+HP": "#d62728",
        "None": "#9467bd",
    }


def plot_recall(classification_report_df: pd.DataFrame, title: str) -> None:
    plt.figure(figsize=(10, 6))
    sns.barplot(x=classification_report_df.index[:-3], y=classification_report_df["recall"][:-3], palette="viridis")
    plt.title(title)
    plt.xlabel("Labels")
    plt.ylabel("Recall")
    plt.xticks(rotation=45, ha="right")
    plt.show()


def plot_confusion_matrix(y_true: Iterable[str], y_pred: Iterable[str], title: str) -> None:
    labels = sorted(set(y_true) | set(y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.show()

def compare_model_with_different_synthetic_training_sizes(
    results_dict: Dict[str, Dict[str, Dict[str, float]]],
    performance_results_real_data: Dict[str, Dict[str, float]],
    save_path: str = "figures/performance_comparison.png",
) -> None:
    plt.style.use("default")
    sns.set_palette("husl")

    class_name_mapping = {
        "EV_NoPV": "EV",
        "NONE": "None",
        "Only_PV": "PV",
        "PV+HP": "PV+HP",
        "EV+PV": "EV+PV",
    }

    def rename_classes_in_results(results: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
        renamed_per_class_f1 = {}
        renamed_per_class_precision = {}
        renamed_per_class_recall = {}
        for old_name, f1_score in results["per_class_f1"].items():
            new_name = class_name_mapping.get(old_name, old_name)
            renamed_per_class_f1[new_name] = f1_score
        for old_name, precision_score in results["per_class_precision"].items():
            new_name = class_name_mapping.get(old_name, old_name)
            renamed_per_class_precision[new_name] = precision_score
        for old_name, recall_score in results["per_class_recall"].items():
            new_name = class_name_mapping.get(old_name, old_name)
            renamed_per_class_recall[new_name] = recall_score
        results["per_class_f1"] = renamed_per_class_f1
        results["per_class_precision"] = renamed_per_class_precision
        results["per_class_recall"] = renamed_per_class_recall
        return results

    synthetic_results = {key: rename_classes_in_results(value) for key, value in results_dict.items()}
    synthetic_keys_sorted = sorted(synthetic_results.keys(), key=lambda x: int(x))

    real_results = rename_classes_in_results(performance_results_real_data)
    data_labels = [f"{int(size):,}" for size in synthetic_keys_sorted]

    all_classes = set()
    for results in synthetic_results.values():
        all_classes.update(results["per_class_f1"].keys())
    all_classes = sorted(list(all_classes))

    consistent_colors = get_consistent_color_mapping()
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for class_name in all_classes:
        class_f1_scores = []
        for size in synthetic_keys_sorted:
            class_f1_scores.append(synthetic_results[size]["per_class_f1"][class_name])

        class_color = consistent_colors.get(class_name, "#95A5A6")
        class_f1_diff = np.array(class_f1_scores) - np.array(real_results["per_class_f1"][class_name])
        ax.plot(class_f1_diff, color=class_color, linewidth=1, alpha=1, label=class_name)

    ax.set_xlabel("Synthetic data size", fontsize=14)
    ax.set_ylabel("F1-Score difference", fontsize=14)
    ax.set_xticks(np.arange(len(data_labels)))
    ax.set_xticklabels(data_labels, rotation=0, ha="center", fontsize=14)
    ax.legend(loc="lower right", fontsize=18)
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON SUMMARY")
    print("=" * 60)

    if performance_results_real_data:
        print("\nReal Data:")
        print("  Per-Class F1-Scores:")
        for class_name, f1_score in sorted(performance_results_real_data["per_class_f1"].items()):
            print(f"    {class_name}: {f1_score:.3f}")

    for size in synthetic_keys_sorted:
        print(f"\nSynthetic Data {size} profiles:")
        print("  Per-Class F1-Scores:")
        for class_name, f1_score in sorted(synthetic_results[size]["per_class_f1"].items()):
            print(f"    {class_name}: {f1_score:.3f}")


def create_label_distribution_comparison(
    real_labels: Iterable[str],
    predicted_labels: Iterable[str],
    save_path: str = "figures/label_distribution_comparison.svg",
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    class_name_mapping = {
        "EV_NoPV": "EV",
        "NONE": "None",
        "Only_PV": "PV",
        "PV+HP": "PV+HP",
        "EV+PV": "EV+PV",
    }

    real_labels_mapped = pd.Series(real_labels).map(class_name_mapping).fillna(pd.Series(real_labels))
    predicted_labels_mapped = pd.Series(predicted_labels).map(class_name_mapping).fillna(pd.Series(predicted_labels))

    real_counts = real_labels_mapped.value_counts()
    pred_counts = predicted_labels_mapped.value_counts()

    all_categories = sorted(set(real_counts.index) | set(pred_counts.index))
    real_counts = real_counts.reindex(all_categories, fill_value=0)
    pred_counts = pred_counts.reindex(all_categories, fill_value=0)

    real_percentages = (real_counts / real_counts.sum()) * 100
    pred_percentages = (pred_counts / pred_counts.sum()) * 100

    x = np.arange(len(all_categories))
    width = 0.35

    bars_real = ax.bar(
        x - width / 2,
        real_percentages.values,
        width,
        label="Real Data",
        color="#2E86AB",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )
    bars_pred = ax.bar(
        x + width / 2,
        pred_percentages.values,
        width,
        label="Predicted on Synthetic",
        color="#F18F01",
        alpha=0.8,
        edgecolor="black",
        linewidth=1,
    )

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Occurrence of label in data (%)", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(all_categories, rotation=0, ha="center", fontsize=14)
    ax.set_ylim(0, max(real_percentages.max(), pred_percentages.max()) * 1.1)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    for bar, percentage in zip(bars_real, real_percentages.values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    for bar, percentage in zip(bars_pred, pred_percentages.values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.5,
            f"{percentage:.1f}%",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    differences = abs(real_percentages - pred_percentages)
    mean_absolute_difference = differences.mean()

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight", facecolor="white")
    plt.show()

    print("\n" + "=" * 60)
    print("LABEL DISTRIBUTION COMPARISON")
    print("=" * 60)

    print(f"\nReal Data (n={real_counts.sum()}):")
    for category in all_categories:
        count = real_counts[category]
        percentage = real_percentages[category]
        print(f"  {category}: {count:,} ({percentage:.1f}%)")

    print(f"\nPredicted on Synthetic Data (n={pred_counts.sum()}):")
    for category in all_categories:
        count = pred_counts[category]
        percentage = pred_percentages[category]
        print(f"  {category}: {count:,} ({percentage:.1f}%)")

    print(f"\nDistribution Differences:")
    for category in all_categories:
        diff = abs(real_percentages[category] - pred_percentages[category])
        print(f"  {category}: {diff:.1f}% difference")

    print(f"\nOverall Statistics:")
    print(f"  Mean Absolute Difference: {mean_absolute_difference:.1f}%")
    print(f"  Max Difference: {differences.max():.1f}% ({all_categories[differences.argmax()]})")
    print(f"  Min Difference: {differences.min():.1f}% ({all_categories[differences.argmin()]})")

def create_hourly_consumption_comparison(
    real_labels_df: pd.DataFrame,
    real_time_series: pd.DataFrame,
    synthetic_labels_pred: Iterable[str],
    synthetic_time_series: pd.DataFrame,
    save_path: str = "figures/hourly_consumption_comparison.png",
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    class_name_mapping = {
        "EV_NoPV": "EV",
        "NONE": "None",
        "Only_PV": "PV",
        "PV+HP": "PV+HP",
        "EV+PV": "EV+PV",
    }

    if not isinstance(real_time_series.index, pd.DatetimeIndex):
        real_time_series.index = pd.to_datetime(real_time_series.index)
    if not isinstance(synthetic_time_series.index, pd.DatetimeIndex):
        synthetic_time_series.index = pd.to_datetime(synthetic_time_series.index)

    real_labels_mapped = real_labels_df["Category"].map(class_name_mapping).fillna(real_labels_df["Category"])
    synthetic_labels_mapped = pd.Series(synthetic_labels_pred).map(class_name_mapping).fillna(
        pd.Series(synthetic_labels_pred)
    )

    all_categories = sorted(set(real_labels_mapped) | set(synthetic_labels_mapped))
    color_dict = get_consistent_color_mapping()

    for category in all_categories:
        if category not in real_labels_mapped.values:
            continue

        original_category = next((orig for orig, mapped in class_name_mapping.items() if mapped == category), category)
        category_ids = real_labels_df[real_labels_df["Category"] == original_category]["ID"].values
        real_columns = real_time_series.columns

        category_ids_str = [str(id_val) for id_val in category_ids]
        category_ids_int = []
        for id_val in category_ids:
            try:
                category_ids_int.append(int(id_val))
            except Exception:
                pass

        available_ids = []
        available_ids.extend([id_val for id_val in category_ids if id_val in real_columns])
        available_ids.extend([id_val for id_val in category_ids_str if id_val in real_columns])
        available_ids.extend([id_val for id_val in category_ids_int if id_val in real_columns])

        if len(available_ids) == 0:
            try:
                real_columns_int = [int(col) for col in real_columns]
                available_ids = [col for col, col_int in zip(real_columns, real_columns_int) if col_int in category_ids]
            except Exception:
                pass

        available_ids = list(set(available_ids))

        if available_ids:
            category_data = real_time_series[available_ids]
            hourly_mean = category_data.groupby(category_data.index.hour).mean().mean(axis=1)
            category_color = color_dict.get(category, "#95A5A6")
            ax.plot(
                hourly_mean.index,
                hourly_mean.values,
                color=category_color,
                linewidth=2.5,
                linestyle="-",
                label=f"{category} (Real)",
                alpha=0.9,
            )

    for category in all_categories:
        if category not in synthetic_labels_mapped.values:
            continue

        category_mask = synthetic_labels_mapped == category
        category_columns = synthetic_time_series.columns[category_mask.values]

        if len(category_columns) == 0:
            continue

        category_data = synthetic_time_series[category_columns]
        hourly_mean = category_data.groupby(category_data.index.hour).mean().mean(axis=1)
        category_color = color_dict.get(category, "#95A5A6")
        ax.plot(
            hourly_mean.index,
            hourly_mean.values,
            color=category_color,
            linewidth=2.5,
            linestyle="--",
            label=f"{category} (Synthetic)",
            alpha=0.9,
        )

    ax.set_xlabel("Hour of Day", fontsize=14)
    ax.set_ylabel("Mean Consumption (kWh)", fontsize=14)
    ax.set_xticks(range(0, 24, 2))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, 24, 2)], fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    from matplotlib.lines import Line2D

    legend_elements = []
    plotted_categories_real = set()
    plotted_categories_synthetic = set()

    for category in all_categories:
        if category in real_labels_mapped.values:
            plotted_categories_real.add(category)
        if category in synthetic_labels_mapped.values:
            plotted_categories_synthetic.add(category)

    for category in sorted(all_categories):
        if category in color_dict:
            if category in plotted_categories_real:
                legend_elements.append(
                    Line2D([0], [0], color=color_dict[category], linewidth=2.5, linestyle="-", label=f"{category}")
                )

    legend_elements.extend(
        [
            Line2D([0], [0], color="black", linewidth=2, linestyle="-", label=r"$\bf{Real Data}$"),
            Line2D([0], [0], color="black", linewidth=2, linestyle="--", label=r"$\bf{Synthetic Data}$"),
        ]
    )

    ax.legend(handles=legend_elements, loc="upper left", fontsize=14, framealpha=0.9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()

    print(f"\n✅ Hourly consumption comparison saved to: {save_path}")

def create_hourly_load_boxplot_comparison(
    real_df: pd.DataFrame,
    synthetic_df: pd.DataFrame,
    real_labels: pd.DataFrame,
    synthetic_labels: Iterable[str],
    save_path: Optional[str] = None,
) -> None:
    color_map = get_consistent_color_mapping()
    class_name_mapping = {
        "EV_NoPV": "EV",
        "NONE": "None",
        "Only_PV": "PV",
        "PV+HP": "PV+HP",
        "EV+PV": "EV+PV",
    }

    real_labels = real_labels.copy()
    real_labels["ID"] = real_labels["ID"].astype(str)
    syn_labels = pd.DataFrame(synthetic_labels, columns=["Category"])
    syn_labels["ID"] = np.arange(1, len(syn_labels) + 1).astype(str)

    real_df_long = real_df.melt(ignore_index=False, var_name="ID", value_name="Load")
    synthetic_df_long = synthetic_df.melt(ignore_index=False, var_name="ID", value_name="Load")

    real_df_long["Hour"] = real_df_long.index.hour
    synthetic_df_long["Hour"] = synthetic_df_long.index.hour

    real_df_long["ID"] = real_df_long["ID"].astype(str)
    synthetic_df_long["ID"] = synthetic_df_long["ID"].astype(str)
    syn_labels["ID"] = syn_labels["ID"].astype(str)
    real_df_long = real_df_long.merge(real_labels, on="ID", how="left")
    synthetic_df_long = synthetic_df_long.merge(syn_labels, on="ID", how="left")

    real_df_long["Source"] = "Real"
    synthetic_df_long["Source"] = "Synthetic"

    real_df_long["Category"] = real_df_long["Category"].map(class_name_mapping)
    synthetic_df_long["Category"] = synthetic_df_long["Category"].map(class_name_mapping)

    unique_labels_real = [lbl for lbl in real_df_long["Category"].dropna().unique()]
    unique_labels_synthetic = [lbl for lbl in synthetic_df_long["Category"].dropna().unique()]
    all_unique_labels = sorted(set(unique_labels_real + unique_labels_synthetic))
    palette = {label: color_map.get(label, "#999999") for label in all_unique_labels}

    order_labels = [lbl for lbl in class_name_mapping.values() if lbl in all_unique_labels]

    df_combined = pd.concat([real_df_long, synthetic_df_long], axis=0)
    df_combined["Category_Source"] = df_combined["Category"] + "_" + df_combined["Source"]

    combined_palette = {}
    for category in order_labels:
        base_color = palette[category]
        combined_palette[f"{category}_Real"] = base_color
        combined_palette[f"{category}_Synthetic"] = base_color

    combined_order = []
    for category in order_labels:
        combined_order.extend([f"{category}_Real", f"{category}_Synthetic"])

    existing_groups = df_combined["Category_Source"].unique()
    combined_order = [group for group in combined_order if group in existing_groups]

    fig, ax = plt.subplots(figsize=(20, 8))
    sns.boxplot(
        data=df_combined,
        x="Hour",
        y="Load",
        hue="Category_Source",
        palette=combined_palette,
        hue_order=combined_order,
        width=0.6,
        ax=ax,
        showfliers=False,
    )

    for i, patch in enumerate(ax.artists):
        group_name = combined_order[i % len(combined_order)]
        if "_Synthetic" in group_name:
            patch.set_hatch("////")
            patch.set_alpha(0.7)
        else:
            patch.set_alpha(0.7)
        patch.set_edgecolor("black")
        patch.set_linewidth(1)

    if ax.legend_:
        ax.legend_.remove()

    from matplotlib.patches import Patch

    legend_elements = []
    for category in order_labels:
        color = palette[category]
        legend_elements.append(Patch(facecolor=color, edgecolor="black", label=category))

    legend_elements.extend(
        [
            Patch(facecolor="gray", edgecolor="black", label="Real Data"),
            Patch(facecolor="gray", edgecolor="black", hatch="///", label="Synthetic Data"),
        ]
    )

    plt.title("Comparison of Hourly Load Profiles by Category and Source", fontsize=16)
    plt.xlabel("Hour of Day", fontsize=14)
    plt.ylabel("Load [kWh]", fontsize=14)

    ax.legend(handles=legend_elements, title="Categories & Sources", bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=12)

    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)

    plt.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def create_real_data_performance_figure(
    classification_report_dict_real_data: Dict[str, Dict[str, float]],
    classification_report_dict_synthetic_data: Dict[str, Dict[str, Dict[str, float]]],
    save_path: str = "figures/real_data_performance.png",
) -> None:
    plt.style.use("default")
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))

    class_name_mapping = {
        "EV_NoPV": "EV",
        "NONE": "None",
        "Only_PV": "PV",
        "PV+HP": "PV+HP",
        "EV+PV": "EV+PV",
    }

    classes = []
    f1_scores = []
    precisions = []
    recalls = []

    for class_name in classification_report_dict_real_data.keys():
        if class_name not in ["accuracy", "macro avg", "weighted avg"]:
            display_name = class_name_mapping.get(class_name, class_name)
            classes.append(display_name)
            f1_scores.append(classification_report_dict_real_data[class_name]["f1-score"])
            precisions.append(classification_report_dict_real_data[class_name]["precision"])
            recalls.append(classification_report_dict_real_data[class_name]["recall"])

    sorted_data = sorted(zip(classes, f1_scores, precisions, recalls))
    classes, f1_scores, precisions, recalls = zip(*sorted_data)

    x = np.arange(len(classes))
    width = 0.25

    bars_f1 = ax.bar(
        x - width,
        f1_scores,
        width,
        label="F1-Score",
        color="#2E86AB",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars_precision = ax.bar(
        x,
        precisions,
        width,
        label="Precision",
        color="#F18F01",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars_recall = ax.bar(
        x + width,
        recalls,
        width,
        label="Recall",
        color="#C73E1D",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=0, ha="center", fontsize=14)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3, axis="y")

    def add_value_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.01,
                f"{height:.3f}",
                ha="center",
                va="bottom",
                fontweight="bold",
                fontsize=9,
            )

    add_value_labels(bars_f1)
    add_value_labels(bars_precision)
    add_value_labels(bars_recall)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()


def plot_cv_metric_dispersion(
    cv_results_by_experiment: Dict[str, List[Dict[str, float]]],
    save_path: str = "figures/cv_metric_dispersion.svg",
):
    if not cv_results_by_experiment:
        print("⚠️ No cross-validation metrics provided for dispersion plot.")
        return None

    metric_configs = [
        ("macro_f1", "Macro F1-Score", "#2E86AB"),
        ("macro_precision", "Macro Precision", "#F18F01"),
        ("macro_recall", "Macro Recall", "#C73E1D"),
    ]

    filtered_items = []
    for label, fold_metrics in cv_results_by_experiment.items():
        if not fold_metrics:
            print(f"⚠️ No fold metrics available for '{label}'; skipping in dispersion plot.")
            continue
        filtered_items.append((label, fold_metrics))

    if not filtered_items:
        print("⚠️ All cross-validation metric collections were empty; skipping dispersion plot.")
        return None

    plt.style.use("default")
    fig, ax = plt.subplots(figsize=(10, 6))
    experiment_labels = [label for label, _ in filtered_items]
    positions = np.arange(len(experiment_labels))
    num_metrics = len(metric_configs)
    width = 0.2

    legend_handles = []
    for idx, (metric_key, title, color) in enumerate(metric_configs):
        metric_data = []
        for _, fold_metrics in filtered_items:
            values = [record.get(metric_key) for record in fold_metrics if record.get(metric_key) is not None]
            if not values:
                values = [np.nan]
            metric_data.append(values)

        offset = (idx - (num_metrics - 1) / 2) * (width * 1.2)
        box_props = ax.boxplot(
            metric_data,
            positions=positions + offset,
            widths=width,
            patch_artist=True,
        )

        for patch in box_props["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_edgecolor("black")
            patch.set_linewidth(1.0)

        for element in ["medians", "caps", "whiskers"]:
            for artist in box_props[element]:
                if hasattr(artist, "set_color"):
                    artist.set_color(color if element == "medians" else "black")
                if hasattr(artist, "set_linewidth"):
                    artist.set_linewidth(1.2 if element == "medians" else 1.0)

        legend_handles.append(plt.Line2D([0], [0], color=color, linewidth=3, label=title))

    ax.set_xticks(positions)
    ax.set_xticklabels(experiment_labels, rotation=35, ha="right", fontsize=11)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_ylim(0.7, 1.0)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(handles=legend_handles, loc="lower right", fontsize=14)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"✅ Cross-validation dispersion figure saved to: {save_path}")
    return fig

def plot_stat(
    arr_feature_real: np.ndarray,
    arr_feature_synth: np.ndarray,
    ax,
    feature_label: str,
    arr_feature_third: Optional[np.ndarray] = None,
    series_labels: Optional[List[str]] = None,
    descr_font_size: int = 7,
) -> None:
    data_arrays = [arr_feature_real, arr_feature_synth]
    labels = list(series_labels[:2]) if series_labels else ["Real", "Synthetic"]

    if arr_feature_third is not None:
        data_arrays.append(arr_feature_third)
        if series_labels and len(series_labels) >= 3:
            labels.append(series_labels[2])
        else:
            labels.append("Holdout")

    box_dict = ax.boxplot(data_arrays, vert=True, patch_artist=True)
    palette = sns.color_palette("Set2", len(data_arrays))
    for patch, color in zip(box_dict["boxes"], palette):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        patch.set_edgecolor("black")
    ax.set_xticklabels(labels)
    ax.set_title(feature_label, fontweight="bold")
    ax.set_ylabel("Value")
    ax.grid()

    num_boxes = len(data_arrays)
    text_offset = 0.15 if num_boxes == 3 else 0.1

    for idx, box in enumerate(box_dict["boxes"]):
        x_pos = idx + 1
        vertices = box.get_path().vertices
        q1 = vertices[0, 1]
        q3 = vertices[2, 1]
        whiskers = [line.get_ydata()[1] for line in box_dict["whiskers"][idx * 2 : idx * 2 + 2]]
        medians = box_dict["medians"][idx].get_ydata()[0]
        ax.text(x_pos + text_offset, q1, f"Q1: {q1:.2f}", va="center", fontsize=descr_font_size, color="blue")
        ax.text(x_pos + text_offset, q3, f"Q3: {q3:.2f}", va="center", fontsize=descr_font_size, color="blue")
        ax.text(x_pos + text_offset, medians, f"Med: {medians:.2f}", va="center", fontsize=descr_font_size, color="red")
        ax.text(x_pos + text_offset, whiskers[0], f"Min: {whiskers[0]:.2f}", va="center", fontsize=descr_font_size, color="green")
        ax.text(x_pos + text_offset, whiskers[1], f"Max: {whiskers[1]:.2f}", va="center", fontsize=descr_font_size, color="green")


def plot_stats(
    arr_features_real: List[np.ndarray],
    arr_features_synth: List[np.ndarray],
    arr_features_third: Optional[List[np.ndarray]] = None,
    feature_labels: Optional[List[str]] = None,
    series_labels: Optional[List[str]] = None,
    title: str = "Comparison of Statistical Features",
):
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(20, 15.5))
    axes = axes.flatten()
    feature_labels = feature_labels or STAT_FEATURE_LABELS

    for idx, ax in enumerate(axes):
        label = feature_labels[idx] if idx < len(feature_labels) else f"Feature {idx + 1}"
        if arr_features_third is not None:
            plot_stat(
                arr_features_real[idx],
                arr_features_synth[idx],
                ax,
                label,
                arr_features_third[idx],
                series_labels=series_labels,
            )
        else:
            plot_stat(
                arr_features_real[idx],
                arr_features_synth[idx],
                ax,
                label,
                series_labels=series_labels,
            )

    plt.suptitle(title, ha="center", fontsize=16, fontweight="bold")
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


def slice_time_series_by_category(time_series_df: pd.DataFrame, labels_df: pd.DataFrame) -> Dict[str, np.ndarray]:
    if "ID" not in labels_df or "Category" not in labels_df:
        raise ValueError("labels_df must contain 'ID' and 'Category' columns.")

    column_lookup = {str(col): col for col in time_series_df.columns}
    labels_df = labels_df.copy()
    labels_df["ID"] = labels_df["ID"].astype(str)

    category_arrays: Dict[str, np.ndarray] = {}
    for category, ids in labels_df.groupby("Category")["ID"]:
        matching_columns = [column_lookup[id_] for id_ in ids if id_ in column_lookup]
        if not matching_columns:
            print(f"⚠️ No matching columns found in time series for category '{category}'. Skipping.")
            continue
        category_arrays[category] = time_series_df[matching_columns].to_numpy()

    return category_arrays


def compute_statistical_feature_arrays(category_arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    feature_results: Dict[str, np.ndarray] = {}
    for category, arr in category_arrays.items():
        if arr.size == 0:
            print(f"⚠️ Empty array encountered for category '{category}'. Skipping.")
            continue
        feature_results[category] = calc_features(arr, axis=0)
    return feature_results


def create_category_feature_statistics_plots(
    real_time_series: pd.DataFrame,
    real_labels_df: pd.DataFrame,
    synthetic_time_series: pd.DataFrame,
    synthetic_labels_df: pd.DataFrame,
    save_dir: str = "figures",
    scenario_tag: str = "synthetic",
    holdout_time_series: Optional[pd.DataFrame] = None,
    holdout_labels_df: Optional[pd.DataFrame] = None,
    holdout_label: str = "Holdout",
) -> None:
    save_path = pathlib.Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    real_category_arrays = slice_time_series_by_category(real_time_series, real_labels_df)
    synth_category_arrays = slice_time_series_by_category(synthetic_time_series, synthetic_labels_df)
    real_feature_arrays = compute_statistical_feature_arrays(real_category_arrays)
    synth_feature_arrays = compute_statistical_feature_arrays(synth_category_arrays)

    holdout_feature_arrays = None
    holdout_series_labels = None
    if holdout_time_series is not None and holdout_labels_df is not None:
        holdout_category_arrays = slice_time_series_by_category(holdout_time_series, holdout_labels_df)
        holdout_feature_arrays = compute_statistical_feature_arrays(holdout_category_arrays)
        holdout_series_labels = holdout_label

    for category in sorted(real_feature_arrays.keys()):
        real_features = real_feature_arrays.get(category)
        synth_features = synth_feature_arrays.get(category)

        if real_features is None or synth_features is None:
            print(f"⚠️ Missing feature arrays for category '{category}'. Skipping comparison.")
            continue

        third_features = None
        series_labels = [f"Real {category}", f"Synthetic {category}"]
        if holdout_feature_arrays and category in holdout_feature_arrays:
            third_features = holdout_feature_arrays[category]
            series_labels.append(f"{holdout_series_labels} {category}")

        fig = plot_stats(
            real_features,
            synth_features,
            arr_features_third=third_features,
            feature_labels=STAT_FEATURE_LABELS,
            series_labels=series_labels,
            title=f"{category} Statistical Feature Comparison",
        )

        output_path = save_path / f"stat_feature_comparison_{category}_{scenario_tag}.svg"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"✅ Statistical comparison for category '{category}' saved to: {output_path}")
