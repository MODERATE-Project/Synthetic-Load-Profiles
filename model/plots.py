import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os


def plot_losses(df, plotPath):
    df.reset_index().plot(x = 'index', y = ['loss_discriminator_real', 'loss_discriminator_fake', 'loss_generator'], alpha = 0.5, figsize = (12, 5))
    plt.title('Training losses')
    plt.xlabel('epoch × batch_index')
    plt.ylabel('value')
    plt.ylim(0, df[['loss_discriminator_real', 'loss_discriminator_fake', 'loss_generator']].max().max()*1.1)
    plt.savefig(plotPath / 'training_losses.png')
    plt.close();


def compare_distributions(X_real, X_synth, plotPath):
    fig = plt.figure(figsize = (8, 5), facecolor = 'w')
    plt.hist(np.array(X_real).flatten(), bins = 100, alpha = 0.5, label = 'Real', color = 'aqua')
    plt.hist(np.array(X_synth).flatten(), bins = 100, alpha = 0.5, label = 'Synthetic', color = 'hotpink')
    plt.title('Comparison of the distributions of the values')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.tight_layout()
    plt.savefig(plotPath / 'load_distrib.png')
    return fig;


def plot_peaks(X_real, X_synth, plotPath):
    peaksReal = X_real.max(axis = 0)
    peaksSynth = X_synth.max(axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([peaksReal, peaksSynth])
    ax.set_xticklabels(['Real peaks', 'Synthetic peaks'])
    plt.title(f'Comparison of peak values')
    plt.ylabel('Value')
    plt.savefig(plotPath / 'peaks.png')
    return fig;


def plot_means(X_real, X_synth, plotPath):
    meansReal = X_real.mean(axis = 0)
    meansSynth = X_synth.mean(axis = 0)
    fig, ax = plt.subplots(figsize = (8, 5), facecolor = 'w')
    ax.boxplot([meansReal, meansSynth])
    ax.set_xticklabels(['Real means', 'Synthetic means'])
    plt.title(f'Comparison of mean values')
    plt.ylabel('Value')
    plt.savefig(plotPath / 'means.png')
    return fig;


def plot_mean_profiles(X_synth, plotPath):
    plt.figure(figsize = (10, 5))
    sns.heatmap(X_synth.astype(float).mean(axis = 1).reshape(24, -1))
    plt.title('Mean synthetic profile')
    plt.savefig(plotPath / 'mean_synth_profile.png');


def plot_wrapper(X_real, X_synth, runPath, return_ = False):
    plotPath = runPath / 'plots'
    os.makedirs(plotPath) if not os.path.exists(plotPath) else None
    X_synth = X_synth[:, 1:]
    fig_comp = compare_distributions(X_real, X_synth, plotPath)
    fig_peaks = plot_peaks(X_real, X_synth, plotPath)
    fig_means = plot_means(X_real, X_synth, plotPath)
    plot_mean_profiles(X_synth, plotPath)
    if return_:
        return fig_comp, fig_peaks, fig_means


def model_plot_wrapper(X_real, X_synth, plotPath):
    X_synth = X_synth[:, 1:]
    compare_distributions(X_real, X_synth, plotPath)
    plot_peaks(X_real, X_synth, plotPath)
    plot_means(X_real, X_synth, plotPath)
    plot_mean_profiles(X_synth, plotPath)


def plot_similarity_boxplot(similarity_map: dict[str, np.ndarray]) -> plt.Figure:
    """
    Create a publication-ready boxplot comparing cosine similarity distributions.
    """
    if not similarity_map:
        raise ValueError("No similarity data provided for plotting.")

    records = []
    for label, values in similarity_map.items():
        if values is None:
            continue
        arr = np.asarray(values, dtype=np.float64).ravel()
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        records.append(pd.DataFrame({"Dataset": label, "Similarity": arr}))

    if not records:
        raise ValueError("Similarity arrays were empty after removing NaNs.")

    df = pd.concat(records, ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette("colorblind", df["Dataset"].nunique())

    sns.boxplot(
        data=df,
        x="Dataset",
        y="Similarity",
        ax=ax,
        palette=palette,
        width=0.6,
        showcaps=True,
        boxprops={"linewidth": 1.5},
        medianprops={"linewidth": 2, "color": "black"},
        whiskerprops={"linewidth": 1.5},
        capprops={"linewidth": 1.5},
        fliersize=2,
    )

    ax.set_xlabel("")
    ax.set_ylabel("Cosine similarity", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", labelrotation=15, labelsize=10)
    ax.tick_params(axis="y", labelsize=10)
    ax.grid(axis="y", linestyle="--", linewidth=0.8, alpha=0.7)
    fig.tight_layout()

    return fig
