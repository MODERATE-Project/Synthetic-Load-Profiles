from torch import nn, optim, float32, full, randn
from torch.utils.data import DataLoader
import pandas as pd
import os
from tqdm import tqdm
import torch
import numpy as np
import gzip

from model.data_manip import data_prep_wrapper, invert_min_max_scaler, revert_reshape_arr
from model.layers import layersGen, layersDis
from model.plot import create_plots


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
    def __init__(self, model):
        super(Discriminator, self).__init__()
        self.model = model
    
    def forward(self, data):
        output = self.model(data).flatten()
        return output


class GAN(nn.Module):
    def __init__(
            self,
            dataset,
            params,
            outputPath,
            modelStatePath = None,
            wandb = None,
            useMarimo = False
        ):
        super().__init__()
        self.inputDataset = dataset
        self.dataset, self.dfIdx, self.arr_minMax = \
            data_prep_wrapper(df = self.inputDataset, dayCount = DAY_COUNT, featureRange = FEATURE_RANGE)
        self.params = params
        self.outputPath = outputPath
        self.modelStatePath = modelStatePath
        self.wandb = wandb
        self.useMarimo = useMarimo

        # Get parameters from `params.py`
        for key, value in self.params.items():
            setattr(self, key, value)

        # Get layers from `layers.py`
        self.layersGen, self.layersDis = layersGen, layersDis

        # Create DataLoader object
        self.dataLoader = \
            DataLoader(dataset = self.dataset, batch_size = self.batchSize, shuffle = True)

        # Initialize generator and discriminator
        self.Gen = Generator(model = self.layersGen).to(device = self.device)
        self.Dis = Discriminator(model = self.layersDis).to(device = self.device)
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
                self.modelState = torch.load(file)
            if not (self.dfIdx == self.modelState['dfIdx']).all():
                raise ValueError('Timestamps do not match!')
            
            # Adjust min and max values
            self.arr_minMaxOld = self.modelState['minMax']
            self.arr_minMax = np.array([min(self.arr_minMax[0], self.arr_minMaxOld[0]), max(self.arr_minMax[1], self.arr_minMaxOld[1])])

            # Adjust generator and discriminator
            self.Gen.load_state_dict(self.modelState['gen_state_dict'])
            self.Dis.load_state_dict(self.modelState['dis_state_dict'])
            self.optimGen.load_state_dict(self.modelState['optim_gen_state_dict'])
            self.optimDis.load_state_dict(self.modelState['optim_dis_state_dict'])


    def train(self):
        logs = []

        # Create progress bar
        if not self.useMarimo:
            progress = tqdm(range(self.epochCount))
        else:
            import marimo as mo
            progress = mo.status.progress_bar(range(self.epochCount))
        
        for epoch in progress:
            loss_dict = {'_Dis.loss.real': 0, '_Dis.loss.fake': 0, '_Gen.loss': 0}
            gradDis_dict, gradGen_dict = {}, {}

            # Deactivate dropout layers in generator
            if epoch == self.dropoutOffEpoch:
                for module in self.Gen.modules():
                    if isinstance(module, nn.Dropout2d):
                        module.p = 0
            
            for batchIdx, data in enumerate(self.dataLoader):
                xReal = data.to(device = self.device, dtype = float32)

                # Create label vectors
                labelsReal = full(size = (xReal.shape[0],), fill_value = self.labelReal, dtype = float32, device = self.device)
                labelsFake = full(size = (xReal.shape[0],), fill_value = self.labelFake, dtype = float32, device = self.device)

                # Train discriminator with real data
                self.Dis.zero_grad()
                yReal = self.Dis(xReal)
                lossDisReal = self.lossFct(yReal, labelsReal)
                lossDisReal.backward()

                # Train discriminator with fake data
                noise = randn(xReal.shape[0], self.dimNoise, 1, 1, device = self.device)
                xFake = self.Gen(noise)
                yFake = self.Dis(xFake.detach())
                lossDisFake = self.lossFct(yFake, labelsFake)
                lossDisFake.backward()

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
                self.optimDis.step()

                # Train generator
                for idx in range(self.genLoopCount):
                    self.Gen.zero_grad()
                    noise = randn(xReal.shape[0], self.dimNoise, 1, 1, device = self.device)
                    xFake = self.Gen(noise)
                    yFakeNew = self.Dis(xFake)
                    lossGen = self.lossFct(yFakeNew, labelsReal).clone()
                    lossGen.backward(retain_graph = True if idx < self.genLoopCount - 1 else False)
                
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
                self.optimGen.step()

                # Update logged losses
                loss_dict['_Dis.loss.real'] += lossDisReal.cpu().item()
                loss_dict['_Dis.loss.fake'] += lossDisFake.cpu().item()
                loss_dict['_Gen.loss'] += lossGen.cpu().item()

            # Log progress (average losses and gradient norms per epoch)
            log_dict = {key: value/len(self.dataLoader) for key, value in loss_dict.items()} | \
            {key: value/len(self.dataLoader) for key, value in gradDis_dict.items()} | \
            {key: value/len(self.dataLoader) for key, value in gradGen_dict.items()}

            # Log progress with wandb 
            if self.wandb:
                self.wandb.log(log_dict)
            
            # Log progress offline
            logs.append({'epoch': epoch} | log_dict)

            # Export (interim) results
            if (epoch + 1) % self.saveFreq == 0 or epoch + 1 == self.epochCount:
                sampleTemp = self.generate_data()
                
                # Save models
                if self.saveModels or epoch + 1 == self.epochCount:
                    epochModelPath = self.modelPath / f'epoch_{epoch + 1}'
                    os.makedirs(epochModelPath)
                    self.save_model_state(epoch, epochModelPath)

                # Save plots
                if self.savePlots or epoch + 1 == self.epochCount:
                    epochPlotPath = self.plotPath / f'epoch_{epoch + 1}'
                    os.makedirs(epochPlotPath)
                    create_plots(self.inputDataset, sampleTemp, epochPlotPath)
                
                # Save samples
                if self.saveSamples or epoch + 1 == self.epochCount:
                    epochSamplePath = self.samplePath / f'epoch_{epoch + 1}'
                    os.makedirs(epochSamplePath)
                    export_synthetic_data(sampleTemp, epochSamplePath, self.outputFormat)

        # Save logged parameters
        df_log = pd.DataFrame(logs)
        df_log.to_csv(self.logPath / 'log.csv', index = False)

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
        with gzip.open(path / f'epoch_{epoch + 1}.pt.gz', 'wb') as file:
            torch.save(model, file)

    def generate_data(self):
        noise = randn(self.dataset.shape[0], self.dimNoise, 1, 1, device = self.device)
        xSynth = self.Gen(noise)
        xSynth = xSynth.cpu().detach().numpy()
        xSynth = invert_min_max_scaler(xSynth, self.arr_minMax, FEATURE_RANGE)
        xSynth = revert_reshape_arr(xSynth)
        idx = self.dfIdx[:self.dfIdx.get_loc('#####0')]
        xSynth = xSynth[:len(idx)]
        xSynth = np.append(idx.to_numpy().reshape(-1, 1), xSynth, axis = 1)
        return xSynth


def generate_data_from_saved_model(modelStatePath):
    with gzip.open(modelStatePath, 'rb') as file:
        modelState = torch.load(file)
    Gen = Generator(modelState['gen_layers'])
    Gen.load_state_dict(modelState['gen_state_dict'])
    noise = randn(modelState['profileCount'], modelState['dimNoise'], 1, 1, device = modelState['device'])
    xSynth = Gen(noise)
    xSynth = xSynth.cpu().detach().numpy()
    xSynth = invert_min_max_scaler(xSynth, modelState['minMax'], FEATURE_RANGE)
    xSynth = revert_reshape_arr(xSynth)
    idx = modelState['dfIdx'][:modelState['dfIdx'].get_loc('#####0')]
    xSynth = xSynth[:len(idx)]
    xSynth = np.append(idx.to_numpy().reshape(-1, 1), xSynth, axis = 1)
    return xSynth


def export_synthetic_data(arr, outputPath, fileFormat, filename = 'example_synth_profiles'):
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