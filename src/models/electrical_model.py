import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

# Define the PyTorch model
class ElectricalTimeSeriesModel(pl.LightningModule):
    def __init__(self, input_size = 8, output_size = 4, learning_rate=1e-3, n_timesteps = 10):
        super(ElectricalTimeSeriesModel, self).__init__()
        self.learning_rate = learning_rate
        
        # Define the architecture
        self.model = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, output_size),
            nn.Sigmoid(),
        )
        
        # Loss function
        self.loss_fn = nn.MSELoss()
        self.n_timesteps = n_timesteps
    
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        X_batch, y_batch = batch
        
        loss = 0.0
        
        pred = self.model(X_batch)
        loss+= self.loss_fn(pred, y_batch)
        
        self.log("loss", loss, on_step = True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer

    @torch.no_grad()
    def step(self, state, action):
        
        input_state = torch.tensor(np.concatenate([state, action])).float()
        output = self(input_state).numpy()
        return output