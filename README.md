
# Synthetic Load Profile GAN

In this project, a GAN for synthesizing electricity load profiles was developed.


## Getting started

#### Prerequisites

The recommended Python version for running this code is 3.11.

#### Installation

1) Clone the repository to your local machine:

```sh
git clone https://github.com/MODERATE-Project/GAN.git
```

2) Navigate to the project directory:

```sh
cd path/to/repository/GAN
```

3) Create an enviroment:
   
    Conda command for creating a suitable environment (replace *myenv* with the desired enviroment name):

```sh
conda create --name myenv python=3.11
```

4) Activate the enviroment:
   
   Conda command for activating the created enviroment (replace *myenv* with the selected name):

```sh
conda activate myenv
```

5) Install required Python packages:

```sh
conda install pip
```

```sh
pip install -r requirements.txt
```


## Preparing the input data

The input data needs to be provided in form of a CSV file.

The data should roughly cover one year (min 365, max 368 days) of hourly electricity consumption values.

Each column of the CSV file should correspond to a single profile/household.

The first column of the CSV file needs to be an index column (ideally containing timestamps).

Example:

![Example_CSV_structure](/readme/Example_CSV_structure.png)


## Creating a project and running the GAN to train a model

There are two ways to run the code:

#### 1. Marimo notebook

A marimo notebook is provided for easily uploading files, creating projects and training models.

The notebook can be accessed by running the following command in the project directory:

```sh
marimo run marimo.py
```

After uploading the required Input file(s) and adjusting the settings, the program can be started by pressing the "Start" button below the options menu.

<span style='color:red'>**⚠ marimo notebooks only allow file sizes up to 100 MB; for larger input files, the Python script has to be used ⚠**</span>

#### 2. Python script

As an alternative to the marimo notebook, a Python script ("run.py") can be used to create projects and train models.

Settings have to be adjusted directly in the script and file paths have to be provided for the input files. Advanced options are not provided here, however, a multitude of underlying parameters can be adjusted in "model" → "GAN_params.py" and "WGAN_params.py".


## Training parameters

The following (hyper)parameters can be adjusted:

* <ins>model type</ins>: Lets you choose between an ordinary GAN and a WGAN-GP model.
* <ins>output format</ins>: Lets you choose between three possible file formats for the synthetic data: ".npy", ".csv" and ".xlsx".
* <ins>log metric</ins>: Whether or not to log a composite metric for checking the quality of the results in every epoch.
* <ins>use Wandb</ins>: Whether or not to track certain parameters online.
* <ins>epochCount/Number of epochs</ins>: Amount of epochs for training.
* <ins>save frequency</ins>: Defines the frequency of epochs at which results should be saved.
* <ins>save models</ins>: Whether or not to save models at the specified frequency.
* <ins>save plots</ins>: Whether or not to save plots at the specified frequency.
* <ins>save samples</ins>: Whether or not to save samples at the specified frequency.
* <ins>check for min stats</ins>: Epoch after which the model checks whether the composite metric improves before deciding whether to plot the results.
* <ins>batch size</ins>: The batch size (number of training examples processed together in one forward and backward pass) used for training.
* <ins>device</ins>: Lets you choose between CPU and GPUs for creating and training a model. Leave the default value to enable automatic GPU detection.
* <ins>loss function</ins>: Loss function used for the generator and the discriminator in the ordinary GAN. By default, the binary cross entropy loss function (BCELoss) is used. If another loss function is chosen, additional adaptions to the code might be needed.
* <ins>lrGen/lrDis</ins>: Define the learning rates of the generator and the discriminator.
* <ins>betas</ins>: By default, AdamOptimizor is used in both the Generator and Discriminator. The beta values define the moving averages.
* <ins>genLoopCount</ins>: When a model is trained, in the beginning, the discriminator might outperform the generator, leading to no training effect. The generator can be trained multiple times per iteration, defined by this variable.
* <ins>dropoutOffEpoch</ins>: Defines the epoch after which all dropout layers in the generator are deactivated (might imporve the results).