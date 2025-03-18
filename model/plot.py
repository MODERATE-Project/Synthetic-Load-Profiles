import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from pathlib import Path
from numba import njit
import re
import json


features = ['means', 'standard deviations', 'minima', 'maxima', 'medians', 'skews', 'peak to peak ranges', 'lower quartiles', 'upper quartiles']
#Order must align with `calc_features`!

########################################################################################################################

def plot_distrib(arr_real, arr_synth):
    fig = plt.figure(figsize = (7, 5))
    plt.hist(arr_real.flatten(), bins = 100, alpha = 0.5, label = 'Real', color = 'aqua')
    plt.hist(arr_synth.flatten(), bins = 100, alpha = 0.5, label = 'Synthetic', color = 'hotpink')
    plt.title('Value distributions', fontweight = 'bold')
    plt.xlabel('electricity consumption [kW]', fontweight = 'bold')
    plt.ylabel('frequency of values occuring', fontweight = 'bold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.close()
    return fig

########################################################################################################################

def plot_stat(arr_featureReal, arr_featureSynth, ax, title, descrFontSize = 7):
    box_dict = ax.boxplot([arr_featureReal, arr_featureSynth], vert = True)
    ax.set_xticklabels(['real', 'synthetic'])
    ax.set_title(title, fontweight = 'bold')
    ax.set_ylabel('value')
    ax.grid()
    for idx, box in enumerate(box_dict['boxes']):
        x_pos = idx + 1
        q1 = box.get_path().vertices[0, 1]
        q3 = box.get_path().vertices[2, 1]
        whiskers = [line.get_ydata()[1] for line in box_dict['whiskers'][idx*2:idx*2 + 2]]
        medians = box_dict['medians'][idx].get_ydata()[0]
        ax.text(x_pos + 0.1, q1, f'Q1: {q1:.2f}', va = 'center', fontsize = descrFontSize, color = 'blue')
        ax.text(x_pos + 0.1, q3, f'Q3: {q3:.2f}', va = 'center', fontsize = descrFontSize, color = 'blue')
        ax.text(x_pos + 0.1, medians, f'Med: {medians:.2f}', va='center', fontsize = descrFontSize, color='red')
        ax.text(x_pos + 0.1, whiskers[0], f'Min: {whiskers[0]:.2f}', va = 'center', fontsize = descrFontSize, color = 'green')
        ax.text(x_pos + 0.1, whiskers[1], f'Max: {whiskers[1]:.2f}', va = 'center', fontsize = descrFontSize, color = 'green')


def calc_features(arr, axis):
    arr_features = np.array([
        arr.mean(axis = axis),
        arr.std(axis = axis),
        arr.min(axis = axis),
        arr.max(axis = axis),
        np.median(arr, axis = axis),
        skew(arr, axis = axis),
        np.ptp(arr, axis = axis),
        np.percentile(arr, 25, axis = axis),
        np.percentile(arr, 75, axis = axis)
    ])
    return arr_features


def plot_stats(arr_featuresReal, arr_featuresSynth):
    fig, axes = plt.subplots(nrows = 3, ncols = 3, figsize = (20, 15.5))
    axes = axes.flatten()
    for idx, ax in enumerate(axes):
        plot_stat(arr_featuresReal[idx], arr_featuresSynth[idx], ax, features[idx])
    plt.suptitle('Comparison of...', ha = 'center', fontsize = 16, fontweight = 'bold')
    plt.tight_layout()
    plt.close()
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
    plt.close()
    return fig

########################################################################################################################

@njit
def compute_group_stats(X, sortedIdx, boundaries, groupCount, colCount):
    out = np.empty((groupCount, 6), dtype = np.float32)
    for i in range(groupCount):
        start, end = boundaries[i], boundaries[i + 1]
        groupSize = end - start
        meanTotal, stdTotal, minTotal, maxTotal, medianTotal, skewTotal = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        for j in range(colCount):
            n = groupSize
            tmp = np.empty(n, dtype = np.float32)
            for k in range(n):
                tmp[k] = X[sortedIdx[start + k], j]

            # Compute mean, min and max in one pass.
            s = tmp[0]
            min = tmp[0]
            max = tmp[0]
            for k in range(1, n):
                v = tmp[k]
                s += v
                if v < min:
                    min = v
                if v > max:
                    max = v
            mean = s/n

            # Compute standard deviation and the third central moment for skew.
            ssd = 0.0
            scd = 0.0
            for k in range(n):
                diff = tmp[k] - mean
                ssd += diff*diff
                scd += diff*diff*diff
            std = np.sqrt(ssd/n)

            # Compute median via sorting.
            arr = tmp.copy()
            arr.sort()
            if n%2 == 1:
                median = arr[n//2]
            else:
                median = 0.5*(arr[n//2 - 1] + arr[n//2])
            
            # Compute skew; if std is zero, define skew as 0.
            if std == 0:
                skew_val = 0.0
            else:
                skew_val = (scd/n)/(std**3)

            meanTotal += mean
            stdTotal += std
            medianTotal += median
            minTotal += min
            maxTotal += max
            skewTotal += skew_val
        
        out[i, 0] = meanTotal/colCount
        out[i, 1] = stdTotal/colCount
        out[i, 2] = medianTotal/colCount
        out[i, 3] = minTotal/colCount
        out[i, 4] = maxTotal/colCount
        out[i, 5] = skewTotal/colCount
    return out


def compute_trends(arr, arr_dt, res = ['hour', 'day', 'week', 'month']):
    trend_dict = {}
    groups = {}
    if 'hour' in res:
        groups['hour'] = arr_dt.hour.to_numpy()
    if 'day' in res:
        groups['day'] = arr_dt.day.to_numpy()
    if 'week' in res:
        groups['week'] = arr_dt.isocalendar().week.to_numpy()
    if 'month' in res:
        groups['month'] = arr_dt.month.to_numpy()

    for key in res:
        groupKeys = groups[key]
        sortedIdx = np.argsort(groupKeys)
        sortedKeys = groupKeys[sortedIdx]
        boundaries = [0]
        for idx in range(1, len(sortedKeys)):
            if sortedKeys[idx] != sortedKeys[idx - 1]:
                boundaries.append(idx)
        boundaries.append(len(sortedKeys))
        boundaries = np.array(boundaries)
        groupCount = len(boundaries) - 1
        arr_trend = compute_group_stats(arr, sortedIdx, boundaries, groupCount, arr.shape[1])
        trend_dict[key] = arr_trend
    return trend_dict


def plot_mean_trends(trendReal_dict, trendSynth_dict):
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
            axs[idx].set_title(stat)
            if idx == 0:
                axs[idx].legend()
        
        fig.supxlabel('time', fontsize = 12, fontweight = 'bold')
        fig.supylabel('value', fontsize = 12, fontweight = 'bold')
        plt.suptitle(f'{key.capitalize()}ly trend'.replace('Day', 'Dai'), fontweight = 'bold')
        plt.tight_layout()
        plt.close()
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

def histogram_similarity(arr_real, arr_synth, bins = 50):
    arr_flatReal = arr_real.flatten()
    arr_flatSynth = arr_synth.flatten()
    
    # Create histograms with consistent range
    min_ = min(arr_flatReal.min(), arr_flatSynth.min())
    max_ = max(arr_flatReal.max(), arr_flatSynth.max())
    
    histReal, _ = np.histogram(arr_flatReal, bins=bins, range = (min_, max_), density = True)
    histSynth, _ = np.histogram(arr_flatSynth, bins=bins, range = (min_, max_), density = True)
    
    # Compare histograms using RMSE
    return np.sqrt(np.mean((histReal - histSynth)**2))


def composite_metric(arr_real, arr_synth, arr_featuresReal, arr_timeFeaturesReal):
    arr_synth = arr_synth[:, 1:].astype(np.float32)
    
    # Calculate all metrics
    histSimilarity = histogram_similarity(arr_real, arr_synth)
    # Feature normalization for profile features
    arr_featuresSynth = calc_features(arr_synth, axis=0)
    # Store reference values if this is the first run
    if not all(hasattr(composite_metric, attr) for attr in ['feature_means', 'feature_stds_inverted']):
        # Calculate mean and std for each feature type (mean, std, min, max, etc.)
        feature_means = np.mean(arr_featuresReal, axis=1, keepdims=True)  # Shape: (9, 1)
        feature_stds_inverted = 1 / np.std(arr_featuresReal, axis=1, keepdims=True)    # Shape: (9, 1)
        feature_stds_inverted[np.isinf(feature_stds_inverted)] = 1
        
        # Store normalization parameters
        composite_metric.feature_means = feature_means
        composite_metric.feature_stds_inverted = feature_stds_inverted
        
    # Z-score normalize each feature using reference statistics
    normalized_synth_features = (arr_featuresSynth - composite_metric.feature_means) * composite_metric.feature_stds_inverted
     
    synth_feature_stats = np.concatenate([
        np.mean(normalized_synth_features, axis=1),  # Mean of each feature
        np.std(normalized_synth_features, axis=1)    # Std of each feature
    ])

    real_feature_stats = np.ones(shape=synth_feature_stats.shape)
    # MSE between real and synthetic feature statistics (all equally weighted)
    profileFeaturesDistance = np.mean((real_feature_stats - synth_feature_stats)**2)   
    
    # Normalize using reference values from initial run
    reference_values = {
        'hist_similarity': getattr(composite_metric, 'ref_hist_similarity', histSimilarity),
        'profile_features_distance': getattr(composite_metric, 'ref_profile_features', profileFeaturesDistance),
    }
    
    # Store reference values if this is the first run
    if not hasattr(composite_metric, 'ref_hist_similarity'):
        composite_metric.ref_hist_similarity = histSimilarity
        composite_metric.ref_profile_features = profileFeaturesDistance
    
    # Normalize each metric by its reference value
    histSimilarityNorm = histSimilarity/reference_values['hist_similarity']
    profileFeaturesDistanceNorm = profileFeaturesDistance/reference_values['profile_features_distance']
    
    # Combine with appropriate weights
    return 0.5*histSimilarityNorm + 0.5*profileFeaturesDistanceNorm

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