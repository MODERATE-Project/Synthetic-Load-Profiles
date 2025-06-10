from typing import Literal
from torch import nn, optim, float32, full, randn
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import torch
import numpy as np
import gzip
from contextlib import nullcontext
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor
import copy

from model.data_manip import data_prep_wrapper, invert_min_max_scaler, revert_reshape_arr
from model.plot import create_plots, create_html, plot_stats_progress
from model.utils import compute_trends, calc_features, composite_metric


DAY_COUNT = 368
FEATURE_RANGE = (-1, 1)


class Generator(nn.Module):
    def __init__(self, model):
        super(Generator, self).__init__()
        self.model = model
    
    def forward(self, noise):
        output = self.model(noise)
        return output


class Discriminator(nn.Module):
    def __init__(self, model, model_type='GAN'):
        super(Discriminator, self).__init__()
        self.model = model
        self.model_type = model_type
    
    def forward(self, data):
        output = self.model(data).flatten()
        return output


class GAN(nn.Module):
    def __init__(
            self,
            dataset,
            params,
            outputPath,
            modelType: Literal['GAN', 'WGAN'],
            modelStatePath = None,
            logStats = False,
            wandb = None,
            useMarimo = False
        ):
        super().__init__()
        self.inputDataset = dataset
        self.dataset, self.dfIdx, self.arr_minMax = \
            data_prep_wrapper(df = self.inputDataset, dayCount = DAY_COUNT, featureRange = FEATURE_RANGE)
        self.params = params
        self.outputPath = outputPath
        self.modelType = modelType
        if self.modelType in ['GAN', 'WGAN']:
            print(f"Using '{self.modelType}'.")
        else:
            raise ValueError(f"'{self.modelType}' is not a supported model type ['GAN', 'WGAN'].")
        self.modelStatePath = modelStatePath
        self.logStats = logStats
        self.wandb = wandb
        self.useMarimo = useMarimo
        self.last_saved_model_path = None  # Track the last saved model path

        # Get parameters from `params.py`
        for key, value in self.params.items():
            setattr(self, key, value)

        # Override loss function for GAN with BCEWithLogitsLoss for autocast compatibility
        if self.modelType == 'GAN':
            self.lossFct = nn.BCEWithLogitsLoss()

        # Get layers
        if self.modelType == 'GAN':
            from model.GAN_layers import layersGen, layersDis
        elif self.modelType == 'WGAN':
            from model.WGAN_layers import layersGen, layersDis
        self.layersGen, self.layersDis = layersGen, layersDis

        # Create DataLoader object
        self.dataLoader = \
            DataLoader(dataset = self.dataset, batch_size = self.batchSize, shuffle = True)

        # Initialize generator and discriminator
        self.Gen = Generator(model = self.layersGen).to(device = self.device)
        self.Dis = Discriminator(model = self.layersDis, model_type = self.modelType).to(device = self.device)
        self.optimGen = optim.Adam(params = self.Gen.parameters(), lr = self.lrGen, betas = self.betas)
        self.optimDis = optim.Adam(params = self.Dis.parameters(), lr = self.lrDis, betas = self.betas)

        # Create output directories
        self.logPath = self.outputPath / 'logs'
        os.makedirs(self.logPath)
        self.modelPath = self.outputPath / 'models'
        os.makedirs(self.modelPath)
        self.plotPath = self.outputPath / 'plots'
        os.makedirs(self.plotPath)
        self.samplePath = self.outputPath / 'sample_data'
        os.makedirs(self.samplePath)

        # Continue training existing model
        if self.modelStatePath:
            with gzip.open(self.modelStatePath, 'rb') as file:
                self.modelState = torch.load(file, weights_only = False)
            if not (self.dfIdx == self.modelState['dfIdx']).all():
                raise ValueError('Timestamps do not match!')
            print('Successfully imported existing model!')
            
            # Adjust min and max values
            self.arr_minMaxOld = self.modelState['minMax']
            self.arr_minMax = np.array([min(self.arr_minMax[0], self.arr_minMaxOld[0]), max(self.arr_minMax[1], self.arr_minMaxOld[1])])

            # Adjust generator and discriminator
            self.Gen.load_state_dict(self.modelState['gen_state_dict'])
            self.Dis.load_state_dict(self.modelState['dis_state_dict'])
            self.optimGen.load_state_dict(self.modelState['optim_gen_state_dict'])
            self.optimDis.load_state_dict(self.modelState['optim_dis_state_dict'])

        # Prepare real data for plotting
        self.arr_real = self.inputDataset.to_numpy().astype(np.float32)
        self.arr_dt = pd.to_datetime(self.inputDataset.index)
        del self.inputDataset
        self.arr_featuresReal = calc_features(self.arr_real, axis = 0)
        self.arr_timeFeaturesReal = calc_features(self.arr_real, axis = 1)
        self.trendReal_dict = compute_trends(self.arr_real, self.arr_dt)

        # Setup thread pool for background saving
        self.executor = ThreadPoolExecutor(max_workers=3)  # Increased to handle multiple background tasks

    def train(self):
        logs = []
        stats_list = []  #track all RMSE values
        minStat = float(1)  #initialize with 1 as this is the start value in the first epoch (stats are normalize based on first epoch)

        # Set up mixed precision training if on CUDA
        use_amp = self.device.type == 'cuda'
        scaler = torch.cuda.amp.GradScaler() if use_amp else None

        # Create progress bar
        if not self.useMarimo:
            progress = tqdm(range(self.epochCount))
        else:
            import marimo as mo
            progress = mo.status.progress_bar(range(self.epochCount))

        # Training loop
        for epoch in progress:
            if self.modelType == 'GAN':
                loss_dict = {'_Dis.loss.real': 0, '_Dis.loss.fake': 0, '_Gen.loss': 0}
            elif self.modelType == 'WGAN':
                loss_dict = {'_Dis.loss': 0, '_Gen.loss': 0}
            gradDis_dict, gradGen_dict = {}, {}

            # Deactivate dropout layers in generator
            if epoch == self.dropoutOffEpoch:
                for module in self.Gen.modules():
                    if isinstance(module, nn.Dropout2d):
                        module.p = 0
            
            # Batch loop
            for batchIdx, data in enumerate(self.dataLoader):
                xReal = data.to(device = self.device, dtype = float32)

                # Train discriminator
                if self.modelType == 'GAN':
                    # Create label vectors
                    labelsReal = full(size = (xReal.shape[0],), fill_value = self.labelReal, dtype = float32, device = self.device)
                    labelsFake = full(size = (xReal.shape[0],), fill_value = self.labelFake, dtype = float32, device = self.device)

                    # Train discriminator with real data
                    self.Dis.zero_grad()
                    with torch.amp.autocast(device_type='cuda') if use_amp else nullcontext():
                        yReal = self.Dis(xReal)
                        lossDisReal = self.lossFct(yReal, labelsReal)
                    
                    if use_amp:
                        scaler.scale(lossDisReal).backward()
                    else:
                        lossDisReal.backward()

                    # Train discriminator with fake data
                    with torch.amp.autocast(device_type='cuda') if use_amp else nullcontext():
                        noise = randn(xReal.shape[0], self.dimNoise, 1, 1, device = self.device)
                        xFake = self.Gen(noise)
                        yFake = self.Dis(xFake.detach())
                        lossDisFake = self.lossFct(yFake, labelsFake)
                    
                    if use_amp:
                        scaler.scale(lossDisFake).backward()
                    else:
                        lossDisFake.backward()
                
                elif self.modelType == 'WGAN':
                    # Generate fake data
                    with torch.amp.autocast(device_type='cuda') if use_amp else nullcontext():
                        noise = randn(xReal.shape[0], self.dimNoise, 1, 1, device = self.device)
                        xFake = self.Gen(noise).detach()

                        # Compute discriminator loss
                        self.Dis.zero_grad()
                        yReal = self.Dis(xReal)
                        yFake = self.Dis(xFake)
                        lossDis = yFake.mean() - yReal.mean()

                        # Compute gradient penalty
                        GP = self.compute_gradient_penalty(self.Dis, xReal, xFake, self.device)

                        # Add penalty term to loss
                        lossDisTotal = lossDis + self.lambdaGP*GP
                    
                    if use_amp:
                        scaler.scale(lossDisTotal).backward()
                    else:
                        lossDisTotal.backward()

                # Update logged gradients of discriminator
                for moduleName, module in self.Dis.named_modules():
                    for param_name, param in module.named_parameters(recurse = False):
                        if param.grad is not None:
                            key = f"Dis.{moduleName.split('.')[1]}.{type(module).__name__}.{param_name}.gradnorm"
                            if batchIdx == 0:
                                gradDis_dict[key] = param.grad.norm().cpu().item()
                            else:
                                gradDis_dict[key] += param.grad.norm().cpu().item()

                # Update weights and biases of the discriminator
                if use_amp:
                    scaler.step(self.optimDis)
                else:
                    self.optimDis.step()

                # Train generator
                for idx in range(self.genLoopCount):
                    self.Gen.zero_grad()
                    with torch.amp.autocast(device_type='cuda') if use_amp else nullcontext():
                        noise = randn(xReal.shape[0], self.dimNoise, 1, 1, device = self.device)
                        xFake = self.Gen(noise)
                        yFakeNew = self.Dis(xFake)

                        if self.modelType == 'GAN':
                            lossGen = self.lossFct(yFakeNew, labelsReal).clone()
                        elif self.modelType == 'WGAN':
                            lossGen = -yFakeNew.mean()
                    
                    if use_amp:
                        scaler.scale(lossGen).backward()
                    else:
                        lossGen.backward()
                
                    # Update logged gradients of generator
                    if idx == 0:
                        for moduleName, module in self.Gen.named_modules():
                            for param_name, param in module.named_parameters(recurse = False):
                                if param.grad is not None:
                                    key = f"Gen.{moduleName.split('.')[1]}.{type(module).__name__}.{param_name}.gradnorm"
                                    if batchIdx == 0:
                                        gradGen_dict[key] = param.grad.norm().cpu().item()
                                    else:
                                        gradGen_dict[key] += param.grad.norm().cpu().item()

                    # Update weights and biases of the generator
                    if use_amp:
                        scaler.step(self.optimGen)
                        scaler.update()
                    else:
                        self.optimGen.step()

                # Update logged losses
                if self.modelType == 'GAN':
                    loss_dict['_Dis.loss.real'] += lossDisReal.cpu().item()
                    loss_dict['_Dis.loss.fake'] += lossDisFake.cpu().item()
                elif self.modelType == 'WGAN':
                    loss_dict['_Dis.loss'] += lossDisTotal.cpu().item()
                loss_dict['_Gen.loss'] += lossGen.cpu().item()
                
                # Free memory during training
                if batchIdx % 10 == 0 and self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            # Log progress (average losses and gradient norms per epoch)
            log_dict = {key: value/len(self.dataLoader) for key, value in loss_dict.items()} | \
            {key: value/len(self.dataLoader) for key, value in gradDis_dict.items()} | \
            {key: value/len(self.dataLoader) for key, value in gradGen_dict.items()}

            # Log progress with wandb 
            if self.wandb:
                self.wandb.log(log_dict)
            
            # Clean up GPU memory between epochs
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Generate sample for (interim) result export
            if self.logStats or (epoch + 1) % self.saveFreq == 0 or epoch + 1 == self.epochCount:
                sampleTemp = self.generate_data()
                stats = composite_metric(self.arr_real, sampleTemp, self.arr_featuresReal, self.arr_timeFeaturesReal)
                stats_list.append(stats)

                # Update minStat value
                if epoch == self.params['checkForMinStats']:
                    minStat = min(stats_list)
                
                # Check if we have a new best model
                is_best_model = epoch > self.params['checkForMinStats'] and stats < minStat
                if is_best_model:
                    minStat = stats
                
                # Regular interval saving
                if (epoch + 1) % self.saveFreq == 0:
                    # Save models
                    if self.saveModels:
                        epochModelPath = self.modelPath / f'epoch_{epoch + 1}'
                        os.makedirs(epochModelPath, exist_ok=True)
                        
                        self.save_model_state_background(epoch, epochModelPath)
                        # Delete previous model if one exists
                        if self.last_saved_model_path is not None and self.last_saved_model_path.exists():
                            self.executor.submit(self._delete_previous_model, self.last_saved_model_path)
                        self.last_saved_model_path = epochModelPath
                    
                    # Save plots
                    if self.savePlots:
                        epochPlotPath = self.plotPath / f'epoch_{epoch + 1}'
                        os.makedirs(epochPlotPath, exist_ok=True)
                        self.create_plots_background(self.arr_real, self.arr_featuresReal, self.arr_dt, self.trendReal_dict, sampleTemp, epochPlotPath)
                        if epoch + 1 == self.epochCount:
                            create_html(self.plotPath)  # Keep this synchronous as it's only at the end
                    
                    # Save samples
                    if self.saveSamples:
                        epochSamplePath = self.samplePath / f'epoch_{epoch + 1}'
                        os.makedirs(epochSamplePath, exist_ok=True)
                        self.export_synthetic_data_background(sampleTemp, epochSamplePath, self.outputFormat)
                
                # Save best model when we have a new best (logStats mode)
                elif self.logStats and is_best_model:
                    # Save plots for best model
                    if self.savePlots:
                        epochPlotPath = self.plotPath / f'epoch_{epoch + 1}'
                        os.makedirs(epochPlotPath, exist_ok=True)
                        self.create_plots_background(self.arr_real, self.arr_featuresReal, self.arr_dt, self.trendReal_dict, sampleTemp, epochPlotPath)
                    
                    # Save samples for best model
                    if self.saveSamples:
                        epochSamplePath = self.samplePath / f'epoch_{epoch + 1}'
                        os.makedirs(epochSamplePath, exist_ok=True)
                        self.export_synthetic_data_background(sampleTemp, epochSamplePath, self.outputFormat)
                    
                    # Save model for best model
                    if self.saveModels:
                        epochModelPath = self.modelPath / f'epoch_{epoch + 1}'
                        os.makedirs(epochModelPath, exist_ok=True)
                        self.save_model_state_background(epoch, epochModelPath)
                        # Delete previous model if one exists
                        if self.last_saved_model_path is not None and self.last_saved_model_path.exists():
                            self.executor.submit(self._delete_previous_model, self.last_saved_model_path)
                        self.last_saved_model_path = epochModelPath

            # Log progress offline
            logs_dict = {'epoch': epoch} | log_dict if not self.logStats else {'epoch': epoch} | log_dict | {'stats': stats}
            logs.append(logs_dict)

        # Save logged parameters
        df_log = pd.DataFrame(logs)
        df_log.to_csv(self.logPath / 'log.csv', index = False)
        
        # Plot stats progress at the end of training
        if len(stats_list) > 0:
            # Extract epochs for which we have stats values
            stat_epochs = [logs[i]['epoch'] + 1 for i in range(len(logs)) if 'stats' in logs[i]]
            # Plot the stats progress
            if len(stat_epochs) > 0:
                plot_stats_progress(stat_epochs, stats_list, self.plotPath)

        # Clean up thread pool
        self.executor.shutdown(wait=True)

    def compute_gradient_penalty(self, model, xReal, xFake, device):
        """
        Computes the gradient penalty for the WGAN method.
        """
        epsilon = torch.rand(xReal.shape[0], 1, 1, 1, device = device)  #interpolation factor
        interpolated = (epsilon*xReal + (1 - epsilon)*xFake).requires_grad_(True)

        # Compute critic output on interpolated samples
        with torch.amp.autocast(device_type='cuda', enabled=False):  # Use full precision for gradient penalty
            interpolatedOutput = model(interpolated)

        # Compute gradients w.r.t. the interpolated samples
        gradients = torch.autograd.grad(
            outputs = interpolatedOutput,
            inputs = interpolated,
            grad_outputs = torch.ones_like(interpolatedOutput, requires_grad = False),
            create_graph = True,
            retain_graph = True
        )[0]

        # Compute the norm of the gradients
        gradNorm = gradients.view(xReal.shape[0], -1).norm(2, dim = 1)
        GP = torch.mean((gradNorm - 1)**2)
        return GP

    def save_model_state(self, epoch, path):
        model = {
            'epoch': epoch,
            'profileCount': self.dataset.shape[0],
            'dfIdx': self.dfIdx,
            'minMax': self.arr_minMax,
            'gen_layers': self.layersGen,
            'dis_layers': self.layersDis,
            'gen_state_dict': self.Gen.state_dict(),
            'dis_state_dict': self.Dis.state_dict(),
            'optim_gen_state_dict': self.optimGen.state_dict(),
            'optim_dis_state_dict': self.optimDis.state_dict(),
            'continued_from': self.modelStatePath,
            'feature_range': FEATURE_RANGE
        } | self.params
        with gzip.open(path / f'epoch_{epoch + 1}.pt', 'wb') as file:
            torch.save(model, file)

    def save_model_state_background(self, epoch, path):
        """Save model state in a background process to avoid interrupting training."""
        # Copy state dictionaries and move tensors to CPU
        gen_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in self.Gen.state_dict().items()}
        dis_state_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in self.Dis.state_dict().items()}
        
        model = {
            'epoch': epoch,
            'profileCount': self.dataset.shape[0],
            'dfIdx': self.dfIdx,
            'minMax': self.arr_minMax,
            'gen_layers': self.layersGen,
            'dis_layers': self.layersDis,
            'gen_state_dict': gen_state_dict,  # CPU tensors
            'dis_state_dict': dis_state_dict,  # CPU tensors
            'optim_gen_state_dict': self.optimGen.state_dict(),
            'optim_dis_state_dict': self.optimDis.state_dict(),
            'continued_from': self.modelStatePath,
            'feature_range': FEATURE_RANGE
        } | self.params
        
        # Create file path
        filepath = path / f'epoch_{epoch + 1}.pt'
        
        # Submit the save task to the thread pool
        self.executor.submit(self._save_model_to_file, model, filepath)

    def _save_model_to_file(self, model, filepath):
        """Helper function to save model to file from a background thread."""
        try:
            with gzip.open(filepath, 'wb') as file:
                torch.save(model, file)
            print(f"Successfully saved model to {filepath}")
        except Exception as e:
            print(f"Error saving model to {filepath}: {e}")

    def create_plots_background(self, real_data, features_real, dt, trend_real_dict, synth_data, plot_path):
        """Create plots in a background thread to avoid interrupting training."""
        # Create deep copies of numpy arrays to avoid race conditions
        real_data_copy = real_data.copy()
        features_real_copy = copy.deepcopy(features_real)
        dt_copy = dt.copy()
        trend_real_dict_copy = copy.deepcopy(trend_real_dict)
        synth_data_copy = synth_data.copy()
        
        # Submit the plotting task to the thread pool
        self.executor.submit(self._create_plots_in_background, 
                             real_data_copy, 
                             features_real_copy,
                             dt_copy, 
                             trend_real_dict_copy, 
                             synth_data_copy, 
                             plot_path)
                             
    def _create_plots_in_background(self, real_data, features_real, dt, trend_real_dict, synth_data, plot_path):
        """Helper function to create plots in a background thread."""
        create_plots(real_data, features_real, dt, trend_real_dict, synth_data, plot_path)
        
    def export_synthetic_data_background(self, arr, output_path, file_format, filename='example_synth_profiles'):
        """Export synthetic data in a background thread."""
        # Create a copy of the data to avoid race conditions
        arr_copy = arr.copy()
        
        # Submit the export task to the thread pool
        self.executor.submit(self._export_synthetic_data_in_background, 
                            arr_copy, 
                            output_path, 
                            file_format, 
                            filename)
                            
    def _export_synthetic_data_in_background(self, arr, output_path, file_format, filename='example_synth_profiles'):
        """Helper function to export synthetic data in a background thread."""
        export_synthetic_data(arr, output_path, file_format, filename)

    def generate_data(self):
        # Generate data in smaller batches to save memory
        num_batches = (self.dataset.shape[0] + self.params["batchSize"] - 1) // self.params["batchSize"]
        xSynth_list = []
        with torch.no_grad():  # No need to track gradients during generation
            for i in range(num_batches):
                start_idx = i * self.params["batchSize"]
                end_idx = min((i + 1) * self.params["batchSize"], self.dataset.shape[0])
                current_batch_size = end_idx - start_idx
                
                noise = randn(current_batch_size, self.dimNoise, 1, 1, device = self.device)
                xSynth_batch = self.Gen(noise)
                xSynth_batch = xSynth_batch.cpu().numpy()  # Move to CPU immediately
                xSynth_list.append(xSynth_batch)
                
                # Free memory
                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Combine batches
        xSynth = np.vstack(xSynth_list)
        xSynth = invert_min_max_scaler(xSynth, self.arr_minMax, FEATURE_RANGE)
        xSynth = revert_reshape_arr(xSynth)
        idx = self.dfIdx[:self.dfIdx.get_loc('#####0')]
        xSynth = xSynth[:len(idx)]
        xSynth = np.append(idx.to_numpy().reshape(-1, 1), xSynth, axis = 1)
        return xSynth

    def _delete_previous_model(self, model_path):
        """Delete the previous model directory in background."""
        try:
            if os.path.isdir(model_path):
                for file in os.listdir(model_path):
                    file_path = os.path.join(model_path, file)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                os.rmdir(model_path)
                print(f"Deleted previous model at {model_path}")
        except Exception as e:
            print(f"Error deleting previous model: {e}")


def generate_data_from_saved_model(modelStatePath, n_profiles=None):
    try:
        with gzip.open(modelStatePath, 'rb') as file:
            modelState = torch.load(file)
        print(f"Successfully loaded model from {modelStatePath}")
        
        Gen = Generator(modelState['gen_layers'])
        Gen.load_state_dict(modelState['gen_state_dict'])
        
        # Use n_profiles if specified, otherwise use the original profileCount
        profile_count = n_profiles if n_profiles is not None else modelState['profileCount']
        
        # Generate data in smaller batches to save memory
        batch_size = 32  # Adjust this based on your available memory
        num_batches = (profile_count + batch_size - 1) // batch_size
        xSynth_list = []
        
        with torch.no_grad():  # No need to track gradients during generation
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, profile_count)
                current_batch_size = end_idx - start_idx
                
                noise = randn(current_batch_size, modelState['dimNoise'], 1, 1, device = modelState['device'])
                xSynth_batch = Gen(noise)
                xSynth_batch = xSynth_batch.cpu().numpy()  # Move to CPU immediately
                xSynth_list.append(xSynth_batch)
                
                # Free memory
                if modelState['device'].type == 'cuda':
                    torch.cuda.empty_cache()
        
        # Combine batches
        xSynth = np.vstack(xSynth_list)
        xSynth = invert_min_max_scaler(xSynth, modelState['minMax'], FEATURE_RANGE)
        xSynth = revert_reshape_arr(xSynth)
        idx = modelState['dfIdx'][:modelState['dfIdx'].get_loc('#####0')]
        xSynth = xSynth[:len(idx)]
        xSynth = np.append(idx.to_numpy().reshape(-1, 1), xSynth, axis = 1)
        return xSynth
    except Exception as e:
        print(f"Error loading model from {modelStatePath}: {e}")
        raise


def export_synthetic_data(arr, outputPath, filename = 'example_synth_profiles', fileFormat = ".csv"):
    filePath = outputPath / f'{filename}{fileFormat}'
    fileNewIdx = 2
    while filePath.is_file():
        filePath = outputPath / f'{filename}_{fileNewIdx}{fileFormat}'
        fileNewIdx += 1
    match fileFormat:
        case '.npy':
            np.save(file = filePath, arr = arr)
        case '.csv':
            pd.DataFrame(arr).set_index(0).astype(np.float32).to_csv(filePath)
        case '.xlsx':
            pd.DataFrame(arr).set_index(0).astype(np.float32).to_excel(filePath)