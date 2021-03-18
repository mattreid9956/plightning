import os
import json
from argparse import ArgumentParser
from sklearn.datasets import make_regression
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import pytorch_lightning as pl
# Use bolts just to grab datamodule wrapper
from pl_bolts.datamodules import SklearnDataModule

from plmodel import LinearRegression
import utils


def generate_dataset(n_samples:int, input_dim: int = 40, dtype:str = 'linear', 
                     random_state: np.random.RandomState = 0, **kw):
    valid_dtypes = ['linear', 'non-linear']
    assert dtype in valid_dtypes, f"{dtype} is not an available dataset type to generate, only {valid_dtypes}"
    kw['coef'] = True
    X, y, coef = make_regression(n_samples, input_dim, random_state=random_state, **kw)
    X = X.astype(np.float32)
    y = y.astype(np.float32) 
    if dtype == "non-linear":
        y = np.dot(X**2, coef) + np.random.randn(n_samples)*0.5
        #y = np.expm1((y + abs(y.min())) / 200)
        #y = y**2
        print("Making a non-linear dataset!!!!")
    #y_trans = np.log1p(y)
    return X, y, coef


def cli_main(args, name: str = 'deep_lob'):
    
    pl.seed_everything(1234)    

    # Create some toy data
    input_dim = args.input_dim
    n_samples = args.nsamples
    print(f'feature dimension: ({n_samples}, {input_dim})')
    rng = np.random.RandomState(0)
    print(f"Dataset name: {args.dataset_name}")
    X, y, coef = generate_dataset(n_samples, input_dim, random_state=rng, dtype=args.dataset_name)
    coef = pd.Series(coef)
    y = y.reshape(-1, 1)    
    
    # Create the lightning datamodule
    dm_kwargs = dict(
        X=X, y=y, val_split=0.2, test_split=0.1, 
        num_workers=4, random_state=rng, shuffle=True, 
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
    #tensorboard_dir = os.path.join(outputdata_dir, 'tensorboard')
    tensorboard_dir = "/opt/ml/output/tensorboard"
    print(tensorboard_dir, os.path.exists(tensorboard_dir))

    checkpoint_dir = "/opt/ml/checkpoints" #os.path.join(outputdata_dir, 'checkpoint')
    print(checkpoint_dir, os.path.exists(checkpoint_dir))
    has_checkpoints = False
    if os.path.exists(checkpoint_dir):
        has_checkpoints = len(os.listdir(checkpoint_dir))>0
    
    logger = pl.loggers.TensorBoardLogger(tensorboard_dir)
    es_cb = pl.callbacks.EarlyStopping(
        monitor="val_loss", 
        min_delta=0., 
        patience=3, 
        verbose=False, 
        mode="min"
    )

    mc_cb = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        mode="min",
        dirpath=checkpoint_dir,
        filename=name+'-{epoch:02d}-{val_loss:.6f}',
        #save_weights_only=True,
        save_top_k=5,
    )

    lr_cb = pl.callbacks.LearningRateMonitor()
    gpu_cb = pl.callbacks.GPUStatsMonitor(intra_step_time=True, inter_step_time=True)

    callbacks = [lr_cb, mc_cb, es_cb]
    if torch.cuda.is_available():
        callbacks += [gpu_cb]

    # Trainer    
    trainer_kwargs = dict( 
        gpus=int(os.environ.get('SM_NUM_GPUS',0)),
        default_root_dir=outputdata_dir,
        progress_bar_refresh_rate=10,
        logger=logger,
        callbacks=callbacks,
    )
    if args.resume or has_checkpoints:
        resume_from_checkpoint = utils.find_model_path(mc_cb, "epoch")  # This should be a flag really...
        if os.path.exists(resume_from_checkpoint):
            print(f"INFO: Loading latest checkpoint from {resume_from_checkpoint}")
            # model = AlphaModule.load_from_checkpoint(resume_from_checkpoint)
            trainer_kwargs["resume_from_checkpoint"] = resume_from_checkpoint
        else:
            raise IOError(
                f"WTF you should have checkpoints to load from {resume_from_checkpoint}"
            )
    print(trainer_kwargs)
    trainer = pl.Trainer.from_argparse_args(args, **trainer_kwargs)
    tune_result = trainer.tune(model)
    if tune_result and trainer.model.hparams.lr is None:
        trainer.model.hparams.lr = 1.e-2
    # Fit the model
    result = trainer.fit(model)
    print(result)
    
    test = trainer.test(model)
    print(test)

    y_pred = model(torch.from_numpy(X).cuda()).cpu().detach().numpy()
    residual = pd.Series(y.flatten() - y_pred.flatten())
    f, ax = plt.subplots(1, 1, figsize=(14,10))
    residual.pipe(lambda x:(x - x.mean())/x.std()).plot(kind='hist', bins=25, ax=ax)
    f.savefig(os.path.join(args.model_dir, 'residual.png'))
    
    if input_dim == 1:
        f, ax = plt.subplots(1, 1, figsize=(14,10))
        data = pd.DataFrame(dict(x=X.flatten(), y=y.flatten(), y_pred=y_pred.flatten())).set_index('x').sort_index()
        data[['y', 'y_pred']].plot(ax=ax)
        f.savefig(os.path.join(args.model_dir, 'prediction.png'))
    #fit_result = pd.DataFrame(dict(truth=coef, fitted=list(model.parameters())[0].detach().cpu().numpy()[0]))
    #print(fit_result.sort_values('truth', ascending=False).round(1))
    
    
    # After model has been trained, save its state into model_dir which is then copied to back S3
    if not os.path.exists(args.model_dir):
        os. makedirs(args.model_dir, exist_ok=True)
    with open(os.path.join(args.model_dir, 'model.pth'), 'wb') as f:
        torch.save(model.state_dict(), f)
    
    
if __name__ == '__main__':
  
  resource_config = json.loads(os.environ.get("SM_RESOURCE_CONFIG", "{}"))
  print(os.environ)
    #if len(resource_config)>0:
    # On sagemaker we need to ensure that the path
    #os.environ["NCCL_SOCKET_IFNAME"] = resource_config["network_interface_name"]
    # COUlD CALL IT NODE_RANK OR GROUP_RANK
  if resource_config:
    hosts = resource_config['hosts']
    current_host = resource_config['current_host']
    rank = hosts.index(current_host)
    os.environ['GROUP_RANK'] = str(rank)

  parser = ArgumentParser()
  parser = LinearRegression.add_model_specific_args(parser)
  parser = SklearnDataModule.add_argparse_args(parser)
  parser = pl.Trainer.add_argparse_args(parser)
  
  parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'model_output'))
  #parser.add_argument('--input_dim', type=int, default=40)
  parser.add_argument('--nsamples', type=int, default=100000)
  parser.add_argument('--resume', default=False, action='store_true')
  parser.add_argument('--dataset_name', type=str, default='linear' )

  args = parser.parse_args()
  print(args)
  cli_main(args, name='deep_lob')
  print("Job finished!")
