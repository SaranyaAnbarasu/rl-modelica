import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import DataLoader, TensorDataset

# Define the PyTorch model
class ChillerTimeSeriesModel(pl.LightningModule):
    def __init__(self, input_size, output_size, learning_rate=1e-3, n_timesteps = 10):
        super(ChillerTimeSeriesModel, self).__init__()
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
        
        pred = model(X_batch)
        loss+= self.loss_fn(pred, y_batch)
#         for i in range(self.n_timesteps):
            
#             pred = self()
#             loss += self.loss_fn(
        
        for i in range(X_batch.size(0) - self.n_timesteps):  # For each observation
            predictions = []
            # Initialize prediction with the first timestep
            prediction = self(X_batch[i].unsqueeze(0))  # Forward pass
            predictions.append(prediction)
            
            for t in range(1, self.n_timesteps):
                # Generate predictions for the next timesteps
                input_seq = torch.cat((X_batch[i + t, :4], prediction.squeeze()), dim=0).unsqueeze(0)
                prediction = self(input_seq)
                predictions.append(prediction)
            
            # Compute loss for the sequence of predictions against ground truth
            y_true = y_batch[i:i+self.n_timesteps]
            y_pred = torch.cat(predictions).view(self.n_timesteps, -1)
            loss += self.loss_fn(y_pred, y_true)
        
        self.log("loss", loss, on_step = True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
