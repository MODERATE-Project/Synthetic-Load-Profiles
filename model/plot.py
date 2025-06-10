import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import re
import json
from model.utils import compute_trends, calc_features, composite_metric


features = ['means', 'standard deviations', 'minima', 'maxima', 'medians', 'skews', 'peak to peak ranges', 'lower quartiles', 'upper quartiles']
#Order must align with `calc_features`!

########################################################################################################################

def plot_distrib(arr_real, arr_synth):
    fig = plt.figure(figsize = (7, 5))
    plt.hist(arr_real.flatten(), bins = 100, alpha = 0.5, label = 'Real', color = 'aqua', density = True)
    plt.hist(arr_synth.flatten(), bins = 100, alpha = 0.5, label = 'Synthetic', color = 'hotpink', density = True)
    plt.title('Value distributions', fontweight = 'bold')
    plt.xlabel('electricity consumption [kW]', fontweight = 'bold')
    plt.ylabel('density', fontweight = 'bold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    return fig

########################################################################################################################

def plot_stat(arr_featureReal, arr_featureSynth, ax, title, arr_featureThird=None, descrFontSize = 7):
    # Prepare data arrays and labels
    data_arrays = [arr_featureReal, arr_featureSynth]
    labels = ['train', 'synthetic']
    
    # Add third array if provided
    if arr_featureThird is not None:
        data_arrays.append(arr_featureThird)
        labels.append('holdout')
    
    box_dict = ax.boxplot(data_arrays, vert = True)
    ax.set_xticklabels(labels)
    ax.set_title(title, fontweight = 'bold')
    ax.set_ylabel('value')
    ax.grid()
    
    # Adjust text positioning based on number of boxes
    num_boxes = len(data_arrays)
    text_offset = 0.15 if num_boxes == 3 else 0.1
    
    for idx, box in enumerate(box_dict['boxes']):
        x_pos = idx + 1
        q1 = box.get_path().vertices[0, 1]
        q3 = box.get_path().vertices[2, 1]
        whiskers = [line.get_ydata()[1] for line in box_dict['whiskers'][idx*2:idx*2 + 2]]
        medians = box_dict['medians'][idx].get_ydata()[0]
        ax.text(x_pos + text_offset, q1, f'Q1: {q1:.2f}', va = 'center', fontsize = descrFontSize, color = 'blue')
        ax.text(x_pos + text_offset, q3, f'Q3: {q3:.2f}', va = 'center', fontsize = descrFontSize, color = 'blue')
        ax.text(x_pos + text_offset, medians, f'Med: {medians:.2f}', va='center', fontsize = descrFontSize, color='red')
        ax.text(x_pos + text_offset, whiskers[0], f'Min: {whiskers[0]:.2f}', va = 'center', fontsize = descrFontSize, color = 'green')
        ax.text(x_pos + text_offset, whiskers[1], f'Max: {whiskers[1]:.2f}', va = 'center', fontsize = descrFontSize, color = 'green')



def plot_stats(arr_featuresReal, arr_featuresSynth, arr_featuresThird=None):
    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (20, 15.5))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        if arr_featuresThird is not None:
            plot_stat(arr_featuresReal[idx], arr_featuresSynth[idx], ax, features[idx], arr_featuresThird[idx])
        else:
            plot_stat(arr_featuresReal[idx], arr_featuresSynth[idx], ax, features[idx])
    
    # Update title based on number of datasets
    if arr_featuresThird is not None:
        plt.suptitle('Three-way Comparison of...', ha = 'center', fontsize = 16, fontweight = 'bold')
    else:
        plt.suptitle('Comparison of...', ha = 'center', fontsize = 16, fontweight = 'bold')
    plt.tight_layout()
    return fig

########################################################################################################################

def plot_mean_profiles(arr_real, arr_synth):
    maxCols = min([arr_real.shape[1], arr_synth.shape[1]])
    v_min, v_max = arr_real.mean(axis = 1).min(), arr_real.mean(axis = 1).max()
    fig, axs = plt.subplots(ncols = 3, nrows = 1, figsize = (15, 3))
    sns.heatmap(arr_real.mean(axis = 1).reshape(-1, 24).T, ax = axs[0], vmin=v_min, vmax=v_max)
    sns.heatmap(arr_synth.mean(axis = 1).reshape(-1, 24).T, ax = axs[1], vmin=v_min, vmax=v_max)
    sns.heatmap((arr_synth[:, :maxCols] - arr_real[:, :maxCols]).mean(axis = 1).reshape(-1, 24).T, ax = axs[2])
    axs[0].set_title('Mean real profile', fontweight = 'bold')
    axs[0].tick_params(axis='y', rotation=0)
    axs[1].set_title('Mean synthetic profile', fontweight = 'bold')
    axs[1].tick_params(axis='y', rotation=0)
    axs[2].set_title('Mean difference profile (synthetic - real)', fontweight = 'bold')
    axs[2].tick_params(axis='y', rotation=0)
    plt.tight_layout()
    return fig

########################################################################################################################


def plot_mean_trends(trendReal_dict, trendSynth_dict, trendThird_dict=None):
    trendPlot_dict = {}
    stats = ['mean', 'std', 'median', 'min', 'max', 'skew']
    #Order must align with `compute_group_stats`!
    
    for key in trendReal_dict.keys():
        fig, axs = plt.subplots(2, 3, figsize = (18, 8))
        axs = axs.flatten()
        x = range(1, trendReal_dict[key].shape[0] + 1)

        for idx, stat in enumerate(stats):
            axs[idx].plot(x, trendReal_dict[key][:, idx], label = 'Real', color = 'aqua')
            axs[idx].plot(x, trendSynth_dict[key][:, idx], label = 'Synthetic', color = 'hotpink')
            if trendThird_dict is not None:
                axs[idx].plot(x, trendThird_dict[key][:, idx], label = 'Holdout', color = 'green')
            axs[idx].set_title(stat)
            if idx == 0:
                axs[idx].legend()
        
        fig.supxlabel(key, fontsize = 12, fontweight = 'bold')
        fig.supylabel('value', fontsize = 12, fontweight = 'bold')
        plt.suptitle(f'{key.capitalize()}ly trend'.replace('Day', 'Dai'), fontweight = 'bold')
        plt.tight_layout()
        trendPlot_dict[f'{key}ly_trend'.replace('day', 'dai')] = fig
    return trendPlot_dict

########################################################################################################################

def create_plots(arr_real, arr_featuresReal, arr_dt, trendReal_dict, arr_synth, outputPath = None, createPlots = True, plotTrends = True):
    fig_dict = {}

    arr_synth = arr_synth[:, 1:].astype(np.float32)
    arr_featuresSynth = calc_features(arr_synth, axis = 0)

    if createPlots:
        # Value distributions
        fig_dict['distrib_all_profiles'] = plot_distrib(arr_real, arr_synth)

        # Various statistics
        fig_dict['stats_all_profiles'] = plot_stats(arr_featuresReal, arr_featuresSynth)

        # Mean profiles
        fig_dict['mean_profiles'] = plot_mean_profiles(arr_real, arr_synth)

        if plotTrends:
            # Compute trends
            trendSynth_dict = compute_trends(arr_synth, arr_dt)
            
            # Plot mean trends
            trendPlot_dict = plot_mean_trends(trendReal_dict, trendSynth_dict)
            fig_dict = fig_dict | trendPlot_dict

        for key, value in fig_dict.items():
            value.savefig(outputPath / f'{key}.png', bbox_inches = 'tight', dpi = 100)


########################################################################################################################

def plot_stats_progress(epochs, stats_list, outputPath = None):
    """
    Plot the progress of the combined statistics metric over epochs.
    """
    epochs = np.array(epochs)
    stats_values = np.array(stats_list)
    
    # Find the minimum value and its epoch
    idxMin = np.argmin(stats_values)
    epochMin = epochs[idxMin]
    valueMin = stats_values[idxMin]
    
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    ax.scatter(epochs, stats_values, color = '#1f77b4', alpha = 0.7, label = 'Stats')
    ax.scatter(epochMin, valueMin, color = 'red', s = 100, label = f"Minimum: {valueMin:.4f} at epoch {epochMin}")  #highlights the minimum point     
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
    ax.set_ylabel('Stats Value', fontweight='bold', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle='--', alpha=0.7, axis="y")
    ax.legend(loc="lower left", fontsize=14)
    plt.tight_layout()
    if outputPath:
        plt.savefig(outputPath / 'stats_progress.png', bbox_inches = 'tight', dpi = 150)
    plt.close()


def plot_cdf(train_data, test_data, synthetic_data, save_path=None):
    train_values = train_data.values.flatten()
    test_values = test_data.values.flatten()
    synth_values = synthetic_data.values.flatten()
    # Create separate figure for CDF
    fig = plt.figure(figsize=(8, 6))
    
    # Plot CDFs for all datasets
    x = np.sort(train_values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label='Training Data', linewidth=2, color='blue')
    
    x = np.sort(test_values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label='Test Data', linewidth=2, color='red')
    
    x = np.sort(synth_values)
    y = np.arange(1, len(x) + 1) / len(x)
    plt.plot(x, y, label='Synthetic Data', linewidth=2, color='green')
    
    plt.xlabel('electricity consumption (kWh/h)')
    plt.ylabel('Cumulative Probability')
    plt.legend()
    plt.xlim(0, 2.5)  # Zoom in on x-axis
    
    plt.tight_layout()
    return fig

def analyze_pca_comparison(train_data, test_data, synthetic_data):
    """Analyze and compare PCA projections of training, test and synthetic data."""
    from sklearn.decomposition import PCA
    
    # Set font sizes
    plt.rcParams.update({
        'font.size': 14,
        'axes.titlesize': 16,
        'axes.labelsize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12
    })
    
    # Perform PCA
    pca = PCA(n_components=2)
    # Fit PCA on training data and transform all datasets
    train_pca = pca.fit_transform(train_data.T)
    test_pca = pca.transform(test_data.T)
    synth_pca = pca.transform(synthetic_data.T)
    
    # Create comparison plot
    fig = plt.figure(figsize=(10, 8))
    plt.scatter(train_pca[:, 0], train_pca[:, 1], c='blue', label='Training Data', alpha=0.6)
    plt.scatter(test_pca[:, 0], test_pca[:, 1], c='red', label='Test Data', alpha=0.6)
    plt.scatter(synth_pca[:, 0], synth_pca[:, 1], c='green', label='Synthetic Data', alpha=0.6, marker='x')
    
    plt.title('PCA Comparison of Training, Test and Synthetic Data')
    plt.xlabel('First Principal Component')
    plt.ylabel('Second Principal Component')
    plt.legend()
    
    plt.tight_layout()
    return fig

def plot_wasserstein_by_size(wasserstein_distances: dict):
    """Create a plot of Wasserstein distances by synthetic dataset size."""
    fig = plt.figure(figsize=(10, 6))
    sizes = list(wasserstein_distances.keys())
    wasserstein_dists = [metrics['wasserstein_dist'] for metrics in wasserstein_distances.values()]
    
    plt.plot(sizes, wasserstein_dists, marker='o')
    plt.xlabel('Synthetic dataset size')
    plt.ylabel('Wasserstein Distance')
    plt.tight_layout()
    return fig
    
########################################################################################################################

def create_html(path):
  path =  path
  
  def custom_sort_key(str_):
      # Extract numbers from the string
      numbers = [int(num) for num in re.findall(r'\d+', str_)]
      return (numbers[0] if numbers else float('inf'), str_)
  
  # Get image paths
  imageSets = {}
  plotTypes = ['daily_trend', 'distrib_all_profiles', 'hourly_trend', 'mean_profiles', 'monthly_trend', 'stats_all_profiles', 'weekly_trend']
  for item in plotTypes:
      imageSets[item] = sorted([Path(*imgPath.parts[-2:]) .as_posix() for imgPath in path.rglob('*') if item in str(imgPath) and imgPath.is_file()], key=custom_sort_key)
  imageSets_json = json.dumps(imageSets)

  # Create the HTML content
  HTML = f"""
  <!DOCTYPE html>
  <html>
  <head>
    <meta charset="utf-8">
    <title>Training progression</title>
    <style>
      body {{ font-family: Arial, sans-serif; }}
      #controls {{ text-align: center; margin-top: 20px; }}
      #image-container {{ text-align: center; margin-top: 20px; }}
      input[type="number"] {{ width: 60px; }}
      #slider {{ width: 80%; margin-top: 10px; }}
    </style>
  </head>
  <body>
  <div id="controls">
    <label for="imageSet">Select image type:</label>
    <select id="imageSet">
      {"".join(f'<option value="{k}">{k}</option>' for k in imageSets.keys())}
    </select>
    <br>
    <label for="frequency">Frequency [FPS]:</label>
    <input type="number" id="frequency" value="1" min="0.1" step="0.1">
    <button id="playPause">Play</button>
    <br>
    <input type="range" id="slider" min="0" max="0" value="0">
  </div>
  <div id="image-container">
    <img id="plot" src="" alt="Plot" style="max-width:100%; height:auto;">
    <h2 id="epoch-title"></h2>
  </div>
  <script>
    // Injected imageSets from Python
    var imageSets = {imageSets_json};
    
    // Set the default image set to the first one in the dictionary.
    var defaultSet = Object.keys(imageSets)[0];
    var images = imageSets[defaultSet];
    var currentIndex = 0;
    var interval = null;
    var playing = false;
    var plot = document.getElementById('plot');
    var slider = document.getElementById('slider');
    var playPauseButton = document.getElementById('playPause');
    var frequencyInput = document.getElementById('frequency');
    var imageSetDropdown = document.getElementById('imageSet');
    var epochTitle = document.getElementById('epoch-title');
    
    // Set slider's max based on the selected image set.
    slider.max = images.length - 1;
    
    function updateImage(index) {{
      if (index < 0 || index >= images.length) return;
      currentIndex = index;
      // The image source is constructed relative to the HTML file location.
      plot.src = images[currentIndex];
      slider.value = currentIndex;
      
      // Extract the epoch from the image path using a regex.
      var epochMatch = images[currentIndex].match(/epoch_(\\d+)/);
      if (epochMatch) {{
        epochTitle.textContent = "Epoch: " + epochMatch[1];
      }} else {{
        epochTitle.textContent = "";
      }}
    }}
    
    function playAnimation() {{
      var fps = parseFloat(frequencyInput.value);
      if (isNaN(fps) || fps <= 0) {{
        alert("Please enter a valid frequency (fps) greater than 0.");
        return;
      }}
      var intervalTime = 1000 / fps;
      interval = setInterval(function() {{
        currentIndex = (currentIndex + 1) % images.length;
        updateImage(currentIndex);
      }}, intervalTime);
      playing = true;
      playPauseButton.textContent = "Pause";
    }}
    
    function pauseAnimation() {{
      clearInterval(interval);
      playing = false;
      playPauseButton.textContent = "Play";
    }}
    
    playPauseButton.addEventListener('click', function() {{
      if (!playing) {{
        playAnimation();
      }} else {{
        pauseAnimation();
      }}
    }});
    
    slider.addEventListener('input', function() {{
      pauseAnimation();
      updateImage(parseInt(this.value));
    }});
    
    imageSetDropdown.addEventListener('change', function() {{
      pauseAnimation();
      var selectedSet = this.value;
      images = imageSets[selectedSet];
      currentIndex = 0;
      slider.max = images.length - 1;
      updateImage(0);
    }});
    
    // Display the first image on load.
    updateImage(0);
  </script>
  </body>
  </html>
  """

  # Export HTML file
  with open(path / 'training_progression.html', 'w') as file:
      file.write(HTML)