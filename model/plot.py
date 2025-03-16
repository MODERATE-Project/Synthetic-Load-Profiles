import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import json


statTitle_dict = {
    'mean': 'mean values',
    'std': 'standard deviation values',
    'median': 'median values',
    'min': 'minimum values',
    'max': 'maximum values',
    'skew': 'skew values'
}

########################################################################################################################

def plot_distrib(df_real, df_synth):
    fig = plt.figure(figsize = (7, 5))
    plt.hist(df_real.to_numpy().flatten(), bins = 100, alpha = 0.5, label = 'Real', color = 'aqua')
    plt.hist(df_synth.to_numpy().flatten(), bins = 100, alpha = 0.5, label = 'Synthetic', color = 'hotpink')
    plt.title('Value distributions', fontweight = 'bold')
    plt.xlabel('electricity consumption [kW]', fontweight = 'bold')
    plt.ylabel('frequency of values occuring', fontweight = 'bold')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.close()
    return fig

########################################################################################################################

def plot_stat(statReal, statSynth, ax, title, descrFontSize = 7):
    box_dict = ax.boxplot([statReal, statSynth], vert = True)
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


def calc_skew(data: np.array, axis=0) -> np.array:
    # Use faster, vectorized calculation for skew
    if data.size > 0:  # Avoid division by zero
        m3 = np.mean((data - np.mean(data, axis=axis, keepdims=True))**3, axis=axis)
        m2 = np.mean((data - np.mean(data, axis=axis, keepdims=True))**2, axis=axis)
        # Avoid division by zero
        mask = (m2 == 0)
        skew_values = np.zeros_like(m2)
        valid_indices = ~mask
        if np.any(valid_indices):
            skew_values[valid_indices] = m3[valid_indices] / m2[valid_indices]**(1.5)
    
        return skew_values
    else:
        return np.zeros(data.shape[1])

    
def calc_stats_for_plot(df: pd.DataFrame, stats = ['mean', 'std', 'median', 'min', 'max', 'skew']):
    """Calculate statistics for a dataframe, optimized for performance."""
    # Convert to numpy array once (avoid multiple conversions)
    data = df.to_numpy()
    
    # Pre-allocate results dictionary
    results = {}
    
    # NumPy to be faster
    if 'mean' in stats:
        results['mean'] = np.mean(data, axis=0)
    if 'std' in stats:
        results['std'] = np.std(data, axis=0)
    if 'min' in stats:
        results['min'] = np.min(data, axis=0)
    if 'max' in stats:
        results['max'] = np.max(data, axis=0)
    if 'median' in stats:
        results['median'] = np.median(data, axis=0)
    if 'skew' in stats:
        results['skew'] = calc_skew(data, axis=0)
    
    # Convert back to pandas for consistency with original function
    df_stats = pd.DataFrame(results, index=df.columns)
    return df_stats.T  


def plot_stats(statsReal: pd.DataFrame, statsSynth: pd.DataFrame):
    fig, axes = plt.subplots(nrows = 2, ncols = 3, figsize = (20, 10))
    axes = axes.flatten()
    for stat, ax in zip(statsReal.index, axes):
        plot_stat(statsReal.loc[stat], statsSynth.loc[stat], ax, statTitle_dict[stat])
    plt.suptitle('Comparison of...', ha = 'center', fontsize = 16, fontweight = 'bold')
    plt.tight_layout()
    plt.close()
    return fig


def extract_time_features(data: pd.DataFrame) -> np.array:
    """
    Time-level statistics (across profiles for each time point)
    """
    data_np = data.to_numpy()
    time_features = np.array([
    np.mean(data_np, axis=1),           # Average value per timestep
    np.std(data_np, axis=1),            # Volatility per timestep
    calc_skew(data_np, axis=1),
    np.max(data_np, axis=1),        # Peak values
    np.min(data_np, axis=1),        # Minimum values
    np.median(data_np, axis=1),     # Median values
    np.ptp(data_np, axis=1),        # Peak-to-peak range
    np.percentile(data_np, 25, axis=1),  # Lower quartile
    np.percentile(data_np, 75, axis=1),  # Upper quartile
    ])
    return time_features
    

def extract_profile_features(data: pd.DataFrame) -> np.array:
    """
    Profile-level statistics (across time for each profile)
    """
    # Convert once to numpy for efficiency
    data_np = data.to_numpy()
    profile_features = np.array([
        np.mean(data_np, axis=0),           # Average value per profile
        np.std(data_np, axis=0),            # Volatility per profile
        calc_skew(data_np, axis=0),
        np.max(data_np, axis=0),        # Peak values
        np.min(data_np, axis=0),        # Minimum values
        np.median(data_np, axis=0),     # Median values
        np.ptp(data_np, axis=0),        # Peak-to-peak range
        np.percentile(data_np, 25, axis=0),  # Lower quartile
        np.percentile(data_np, 75, axis=0),  # Upper quartile
    ])
    return profile_features

def histogram_similarity(real_data, synth_data, bins=50):
    # Flatten all data points
    real_flat = real_data.flatten()
    synth_flat = synth_data.flatten()
    
    # Create histograms with consistent range
    data_min = min(real_flat.min(), synth_flat.min())
    data_max = max(real_flat.max(), synth_flat.max())
    
    real_hist, _ = np.histogram(real_flat, bins=bins, range=(data_min, data_max), density=True)
    synth_hist, _ = np.histogram(synth_flat, bins=bins, range=(data_min, data_max), density=True)
    
    # Compare histograms using RMSE
    return np.sqrt(np.mean((real_hist - synth_hist)**2))


def composite_metric(real_data: pd.DataFrame, array_synth: np.array):
    real_data.index = pd.to_datetime(real_data.index)
    real_data = real_data.astype(np.float32)
    df_synth = pd.DataFrame(array_synth).set_index(0).astype(np.float32)
    df_synth.index = pd.to_datetime(df_synth.index)

    r = real_data.to_numpy()
    s = df_synth.to_numpy()
    
    # Calculate all metrics
    hist_similarity = histogram_similarity(r, s)
    profile_features_distance = np.mean((extract_profile_features(real_data) - extract_profile_features(df_synth))**2)
    time_features_distance = np.mean((extract_time_features(real_data) - extract_time_features(df_synth))**2)
    
    # Normalize using reference values from initial run
    reference_values = {
        'hist_similarity': getattr(composite_metric, 'ref_hist_similarity', hist_similarity),
        'profile_features_distance': getattr(composite_metric, 'ref_profile_features', profile_features_distance),
        'time_features_distance': getattr(composite_metric, 'ref_time_features', time_features_distance)

    }
    
    # Store reference values if this is the first run
    if not hasattr(composite_metric, 'ref_stats_rmse'):
        composite_metric.ref_hist_similarity = hist_similarity
        composite_metric.ref_profile_features = profile_features_distance
        composite_metric.ref_time_features = time_features_distance
    
    # Normalize each metric by its reference value
    normalized_hist_similarity = hist_similarity / reference_values['hist_similarity']
    normalized_profile_features = profile_features_distance / reference_values['profile_features_distance']
    normalized_time_features = time_features_distance / reference_values['time_features_distance']
    

    # Combine with appropriate weights
    return  0.4*normalized_hist_similarity + 0.3*normalized_profile_features + 0.3*normalized_time_features

########################################################################################################################

def plot_mean_profiles(df_real, df_synth):
    arr_real = df_real.to_numpy()
    arr_synth = df_synth.to_numpy()
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

def create_df_trend(df_dict, res, stats = ['mean', 'std', 'median', 'min', 'max', 'skew']):
    dfs = []
    for type, df in df_dict.items():
        if res == 'week':
            df.index.week = df.index.isocalendar().week
        df_group = df.groupby([getattr(df.index, res)]).agg(stats)
        df_group = df_group.T.groupby(level = 1).mean()
        df_group.columns = np.arange(df_group.shape[1])
        df_group = df_group.T
        df_group['type'] = type
        df_group['time'] = df_group.index
        dfs.append(df_group)
    df_trend = pd.concat(dfs, axis = 0)
    return df_trend


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

def create_plots(df_real, arr_synth, outputPath = None, createPlots = True):
    fig_dict = {}

    df_real.index = pd.to_datetime(df_real.index)
    df_real = df_real.astype(np.float32)
    df_synth = pd.DataFrame(arr_synth).set_index(0).astype(np.float32)
    df_synth.index = pd.to_datetime(df_synth.index)

    df_statsReal, df_statsSynth = calc_stats_for_plot(df_real), calc_stats_for_plot(df_synth)

    if createPlots:
      # Value distributions
      fig_dict['distrib_all_profiles'] = plot_distrib(df_real, df_synth)

      # Various statistics
      fig_dict['stats_all_profiles'] = plot_stats(df_statsReal, df_statsSynth)

      # Mean profiles
      fig_dict['mean_profiles'] = plot_mean_profiles(df_real, df_synth)

      df_dict = {'Real': df_real, 'Synthetic': df_synth}
      for res in ['hour', 'day', 'week', 'month']:
          df_trend = create_df_trend(df_dict, res)

          # Mean trends
          fig_dict[f'{res}ly_trend'.replace('day', 'dai')] = plot_mean_trends(df_trend, res)

      for key, value in fig_dict.items():
        value.savefig(outputPath / f'{key}.png', bbox_inches = 'tight')
    

########################################################################################################################

def plot_stats_progress(epochs, stats_values, outputPath=None):
    """
    Plot the progress of the combined statistics metric over epochs.
    
    Parameters:
    -----------
    epochs : list or array
        List of epoch numbers
    stats_values : list or array
        List of stats values corresponding to each epoch
    outputPath : Path object, optional
        Path to save the plot
        
    Returns:
    --------
    fig : matplotlib figure
        The generated figure
    """
    # Convert to numpy arrays for easier manipulation
    epochs = np.array(epochs)
    stats_values = np.array(stats_values)
    
    # Find the minimum value and its epoch
    min_idx = np.argmin(stats_values)
    min_epoch = epochs[min_idx]
    min_value = stats_values[min_idx]
    
    # Create the plot
    fig, ax = plt.figure(figsize=(10, 6)), plt.gca()
    
    # Plot all points
    ax.scatter(epochs, stats_values, color='#1f77b4', alpha=0.7, label='Stats')
    
    # Highlight the minimum point
    ax.scatter(min_epoch, min_value, color='red', s=100, 
               label=f'Minimum: {min_value:.4f} at epoch {min_epoch}')
    
    # Add connecting lines for better visualization
    ax.plot(epochs, stats_values, color='#1f77b4', alpha=0.3)
        
    # Set plot labels and title with larger font
    ax.set_xlabel('Epoch', fontweight='bold', fontsize=14)
    ax.set_ylabel('Stats Value', fontweight='bold', fontsize=14)
    
    # Increase tick size
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Set y-axis limits between 0 and 1
    ax.set_ylim(0, 1)
    
    # Add grid for better readability
    ax.grid(True, linestyle='--', alpha=0.7, axis="y")
    
    # Add legend with larger font
    ax.legend(loc="lower left", fontsize=14)
    
    # Tight layout
    plt.tight_layout()
    
    # Save if path is provided
    if outputPath:
        plt.savefig(outputPath / 'stats_progress.png', bbox_inches='tight', dpi=150)
        
    # Close the plot to prevent display in notebook environments
    plt.close()
    
########################################################################################################################

def create_html(path):
  path =  path
  import re
  
  def custom_sort_key(s):
      # Extract numbers from the string
      numbers = [int(num) for num in re.findall(r'\d+', s)]
      return (numbers[0] if numbers else float('inf'), s)
  
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



