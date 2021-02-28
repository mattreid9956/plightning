import os
import sys
import pandas as pd
import numpy as np
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
import pytorch_lightning as pl


class LinearRegression(pl.LightningModule):
    """Linear regression model implementing - with optional L1/L2 regularization
    $$min_{W} ||(Wx + b) - y ||_2^2 $$
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        hidden_dim: int = 60000,
        bias: bool = True,
        learning_rate: float = 1e-4,
        optimizer: torch.optim.Optimizer = torch.optim.Adam,
        l1_strength: float = 0.0,
        l2_strength: float = 0.0,
        **kwargs
    ):
        """
        Args:
            input_dim: number of dimensions of the input (1+)
            output_dim: number of dimensions of the output (default=1)
            bias: If false, will not use $+b$
            learning_rate: learning_rate for the optimizer
            optimizer: the optimizer to use (default='Adam')
            l1_strength: L1 regularization strength (default=None)
            l2_strength: L2 regularization strength (default=None)
        """
        super().__init__()
        self.save_hyperparameters()
        self.optimizer = optimizer
        layers = [
            torch.nn.Linear(in_features=self.hparams.input_dim, 
                      out_features=hidden_dim, 
                      bias=bias)
            torch.nn.Relu(),
            torch.nn.Linear(in_features=hidden_dim, 
                      out_features=self.hparams.output_dim, 
                      bias=bias)
         ]
        self.sequential_module = torch.nn.Sequential(*layers)

    def forward(self, x):
        y_hat = self.sequential_module(x)
        return y_hat

    def training_step(self, batch, batch_idx):
        x, y = batch

        # flatten any input
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y, reduction='sum')

        # L1 regularizer
        if self.hparams.l1_strength > 0:
            l1_reg = sum(param.abs().sum() for param in self.parameters())
            loss += self.hparams.l1_strength * l1_reg

        # L2 regularizer
        if self.hparams.l2_strength > 0:
            l2_reg = sum(param.pow(2).sum() for param in self.parameters())
            loss += self.hparams.l2_strength * l2_reg

        loss /= x.size(0)

        self.log('train_loss', loss, on_epoch=True, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        x = x.view(x.size(0), -1)
        y_hat = self(x)
        loss = F.mse_loss(y_hat, y)
        self.log('val_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.sequential_module.parameters(), lr=self.hparams.learning_rate)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', type=float, default=0.0001)
        parser.add_argument('--input_dim', type=int, default=None)
        parser.add_argument('--output_dim', type=int, default=1)
        parser.add_argument('--bias', default='store_true')
        parser.add_argument('--batch_size', type=int, default=16)
        return parser
