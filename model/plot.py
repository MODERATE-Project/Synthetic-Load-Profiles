import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import skew
from pathlib import Path
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
        arr.ptp(axis = axis),
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

def compute_trends(arr, arr_dt, type_, res = ['hour', 'day', 'week', 'month'], stats = ['mean', 'std', 'min', 'max', 'median', 'skew']):
    dfs = []
    df = pd.DataFrame(arr)
    df.index = arr_dt
    df.index.week = df.index.isocalendar().week
    for item in res:
        df_group = df.groupby([getattr(df.index, item)]).agg(stats)
        df_group = df_group.T.groupby(level = 1).mean()
        df_group.columns = np.arange(df_group.shape[1])
        df_group = df_group.T
        df_group['type'] = type_
        df_group['res'] = item
        df_group['time'] = df_group.index.astype(int)
        dfs.append(df_group)
    return pd.concat(dfs)


def plot_mean_trends(df_trend, res, stats = ['mean', 'std', 'median', 'min', 'max', 'skew']):
    df_trend = df_trend.melt(id_vars = ['type', 'time'], value_vars = stats, var_name = 'statistic', value_name = 'value')
    df_trend['time'] = df_trend['time'].astype(int, errors = 'raise')
    fig = sns.FacetGrid(df_trend, col = 'statistic', col_wrap = 3, sharex = False, sharey = False, height = 3.26, aspect = 1.5)
    fig.map_dataframe(sns.lineplot, x = 'time', y = 'value', hue = 'type', style = 'type')
    fig.set_axis_labels('time')
    fig.axes.flat[0].set_ylabel('value')
    fig.axes.flat[3].set_ylabel('value')
    fig.axes.flat[0].legend()
    plt.suptitle(f'{res.capitalize()}ly trend'.replace('Day', 'Dai'), fontweight = 'bold')
    plt.tight_layout()
    plt.close()
    return fig.fig

########################################################################################################################

def create_plots(arr_real, arr_featuresReal, arr_dt, df_trendReal, arr_synth, outputPath = None, createPlots = True, plotTrends = True):
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
            df_trendSynth = compute_trends(arr_synth, arr_dt, type_ = 'synth')
            df_trend = pd.concat([df_trendReal, df_trendSynth])
            
            # Mean trends
            for res in df_trend['res'].unique():
                fig_dict[f'{res}ly_trend'.replace('day', 'dai')] = plot_mean_trends(df_trend[df_trend['res'] == res], res)

        for key, value in fig_dict.items():
            value.savefig(outputPath / f'{key}.png', bbox_inches = 'tight')

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
    profileFeaturesDistance = np.mean((arr_featuresReal - calc_features(arr_synth, axis = 0))**2)
    timeFeaturesDistance = np.mean((arr_timeFeaturesReal - calc_features(arr_synth, axis = 1))**2)
    
    # Normalize using reference values from initial run
    reference_values = {
        'hist_similarity': getattr(composite_metric, 'ref_hist_similarity', histSimilarity),
        'profile_features_distance': getattr(composite_metric, 'ref_profile_features', profileFeaturesDistance),
        'time_features_distance': getattr(composite_metric, 'ref_time_features', timeFeaturesDistance)
    }
    
    # Store reference values if this is the first run
    if not hasattr(composite_metric, 'ref_stats_rmse'):
        composite_metric.ref_hist_similarity = histSimilarity
        composite_metric.ref_profile_features = profileFeaturesDistance
        composite_metric.ref_time_features = timeFeaturesDistance
    
    # Normalize each metric by its reference value
    histSimilarityNorm = histSimilarity/reference_values['hist_similarity']
    profileFeaturesDistanceNorm = profileFeaturesDistance/reference_values['profile_features_distance']
    timeFeaturesDistanceNorm = timeFeaturesDistance/reference_values['time_features_distance']
    
    # Combine with appropriate weights
    return 0.4*histSimilarityNorm + 0.3*profileFeaturesDistanceNorm + 0.3*timeFeaturesDistanceNorm

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