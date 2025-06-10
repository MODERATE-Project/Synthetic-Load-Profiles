import wandb
from datetime import datetime
from pathlib import Path
import torch
import gc
from model.main import GAN, generate_data_from_saved_model, export_synthetic_data


def run(params, modelType, projectName, train_data, logStats, useWandb, modelPath, createData, useMarimo = False, 
        generate_n_profiles = None, test_data = None):
    if modelPath and createData:
        outputPath = Path(modelPath).parent.parent.parent / 'sample_data' / Path(modelPath).parent.name
        outputPath.mkdir(parents = True, exist_ok = True)
        for i in range(4, 10):
            X_synth = generate_data_from_saved_model(modelPath, n_profiles = generate_n_profiles)
            export_synthetic_data(X_synth, outputPath, filename = f'{generate_n_profiles}_profiles_{i}', fileFormat = ".csv")
    else:
        wandb.init(
            project = 'GAN',
            mode = 'online' if useWandb else 'offline'
        )
        modelName = wandb.run.name
        runNameTSSuffix = datetime.today().strftime('%Y-%m-%d-%H%M%S%f')[:-3]   #added to the end of the run name
        runName = f'{modelName}_{runNameTSSuffix}' if len(modelName) > 0 else runNameTSSuffix
        outputPath = Path().absolute() / 'runs' / projectName / runName
        outputPath.mkdir(parents = True, exist_ok = True)
        # Save the datasets
        if test_data is not None:
            train_data.to_csv(outputPath / 'train_data.csv')
            test_data.to_csv(outputPath / 'test_data.csv')  
        model = GAN(
            dataset = train_data,
            params = params,
            outputPath = outputPath,
            modelType = modelType,
            modelStatePath = modelPath,
            logStats = logStats,
            wandb = wandb,
            useMarimo = useMarimo
        )
        model.train()
        wandb.finish()
        del model
        # Explicitly clean up GPU memory after each run
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Force a synchronization point
            torch.cuda.synchronize()
        # Explicitly trigger garbage collection
        gc.collect()
        