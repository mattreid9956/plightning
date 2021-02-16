import os
from argparse import ArgumentParser
from sklearn.datasets import make_regression
import pandas as pd

import torch
import pytorch_lightning as pl
# Use bolts just to grab datamodule wrapper
from pl_bolts.datamodules import SklearnDataModule

from plmodel import LinearRegression


def cli_main(args, name: str = 'deep_lob'):
    
    pl.seed_everything(1234)    

    # Create some toy data
    input_dim = args.input_dim
    n_samples = args.nsamples
    print(f'feature dimension: ({n_samples}, {input_dim})')
    rng = np.random.RandomState(0)
    X, y, coef = make_regression(n_samples, input_dim, random_state=rng, coef=True)
    coef = pd.Series(coef)
    y = y.reshape(-1, 1)    
    
    # Create the lightning datamodule
    dm_kwargs = dict(
        X=X, y=y, val_split=0.2, test_split=0.1, 
        num_workers=0, random_state=rng, shuffle=False, 
        drop_last=True, pin_memory=True
    )
    dm = SklearnDataModule.from_argparse_args(args, **dm_kwargs)


    # Create the the lightning model
    model_kargs = vars(args)
    model_kargs['input_dim'] = input_dim
    model = LinearRegression(**model_kargs)
    # Magic add the datamodule components, this is just if we want 
    # to run tune() on learning_rate or something later
    model.prepare_data     = dm.prepare_data
    model.setup            = dm.setup
    model.train_dataloader = dm.train_dataloader
    model.val_dataloader   = dm.val_dataloader
    model.test_dataloader  = dm.test_dataloader    
    
    # Configure checkpoints and paths
    outputdata_dir = os.environ.get('SM_OUTPUT_DATA_DIR', 'output')
    tensorboard_dir = os.path.join(outputdata_dir, 'tensorboard')
    print(tensorboard_dir, os.path.exists(tensorboard_dir))

    checkpoint_dir = os.path.join(outputdata_dir, 'checkpoint')
    print(checkpoint_dir, os.path.exists(checkpoint_dir))

    logger = pl.loggers.TensorBoardLogger(tensorboard_dir)
    es_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0., 
        patience=5, 
        verbose=False, 
        mode="min"
    )

    mc_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath=checkpoint_dir,
        filename=name+'-{epoch:02d}-{val_loss:.6f}',
        save_weights_only=True,
        save_top_k=5,
    )

    lr_cb = pl.callbacks.LearningRateMonitor()
    gpu_cb = pl.callbacks.GPUStatsMonitor(intra_step_time=True, inter_step_time=True)

    callbacks = [lr_cb, mc_cb, es_cb]
    if torch.cuda.is_available():
        callbacks += [gpu_cb]

    # Trainer    
    trainer_kwargs = dict( 
        gpus=int(os.environ.get('SM_NUM_GPUS',-1)),
        default_root_dir=outputdata_dir,
        progress_bar_refresh_rate=10,
        logger=logger,
        callbacks=callbacks,
    )
    trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)
    tune_result = trainer.tune(model)
    # Fit the model
    result = trainer.fit(model)
    print(result)

    fit_result = pd.DataFrame(dict(truth=coef, fitted=list(model.parameters())[0].detach().cpu().numpy()[0]))
    print(fit_result.sort_values('truth', ascending=False).round(1))
    
    # After model has been trained, save its state into model_dir which is then copied to back S3
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    
    
if __name__ == '__main__':
  
  resource_config = json.loads(os.environ.get("SM_RESOURCE_CONFIG", "{}"))
  print(os.environ)
  #if len(resource_config)>0:
    # On sagemaker we need to ensure that the path
    #os.environ["NCCL_SOCKET_IFNAME"] = resource_config["network_interface_name"]

  parser = ArgumentParser()
  parser = LinearRegression.add_model_specific_args(parser)
  parser = SklearnDataModule.add_argparse_args(parser)
  parser = pl.Trainer.add_argparse_args(parser)
  
  parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model_output'))
  parser.add_argument('--input-dim', type=int, default=40)
  parser.add_argument('--nsamples', type=int, default=50000)
 
  args = parser.parse_args()
  cli_main(args, name='deep_lob')
  print("Job finished!")
