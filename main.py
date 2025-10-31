import wandb
from datetime import datetime
from pathlib import Path
import torch
import gc
from model.main import GAN, generate_data_from_saved_model, export_synthetic_data
import pandas as pd
from model.plot import plot_similarity_scores

def run(params, modelType, projectName, train_data, logStats, useWandb, modelPath, createData, useMarimo = False, 
        generate_n_profiles = None, test_data = None):
    if modelPath and createData:
        for path in modelPath:
            outputPath_synthetic_data = Path(path).parent / "synthetic_data" 
            name = Path(path).parent.parent.parent.parent.name.replace("WGAN_final_", "").replace("standard", "None").replace("heat pump", "heat_pump")
            outputPath_synthetic_data.mkdir(parents = True, exist_ok = True)
            # load train and test data
            runName = path.parent.parent.parent.name
            data_path = path.parent.parent.parent
            train_data = pd.read_csv(data_path / 'train_data.csv')
            test_data = pd.read_csv(data_path / 'test_data.csv')
            synth_result = generate_data_from_saved_model(
                path, 
                n_profiles = generate_n_profiles,
                train_data = train_data,
                test_data = test_data,
            )

 
            X_synth = synth_result['synthetic_data']
            export_synthetic_data(X_synth, outputPath_synthetic_data, filename = f'{generate_n_profiles}_profiles_{name}', fileFormat = ".parquet")
            # create a report (figures) of the synthetic data
            plot_similarity_scores(
                train_data = train_data,
                test_data = test_data,
                synthetic_before = synth_result['synthetic_data_before'],
                synthetic_after = synth_result['synthetic_data'],
                output_path = data_path / "figures"
            )
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
            modelStatePath = None,
            logStats = logStats,
            wandb = wandb,
            useMarimo = useMarimo
        )
        model.train()
        wandb.finish()
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()    #explicitly clean up GPU memory after each run
            torch.cuda.synchronize()    #force a synchronization point
        gc.collect()    #explicitly trigger garbage collection
