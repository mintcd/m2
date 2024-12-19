import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def create_sequences(data, seq_length):
    sequences = []
    labels = []
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])  # Collect sequences
        labels.append(data[i+seq_length])      # Collect labels
    
    # Convert to tensors
    X = torch.Tensor(np.array(sequences)) # Shape: (num_samples, seq_length)
    y = torch.Tensor(np.array(labels))  # Shape: (num_samples)
    
    # Reshape X to have 3 dimensions: (batch_size, seq_length, input_size)
    X = X.view(X.shape[0], X.shape[1], 1)  # input_size = 1 feature per time step
    
    # Reshape y to have 2 dimensions: (batch_size, output_size)
    y = y.view(-1, 1)  # output_size = 1 target value
    
    return X, y

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, output_size=1, num_layers=2, dropout=0):
        super(LSTMModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Pass the input through the LSTM
        lstm_out, _ = self.lstm(x)

        # We only want the output of the last timestep, hence lstm_out[:, -1, :]
        last_timestep_out = lstm_out[:, -1, :]

        # Pass it through the fully connected layer
        out = self.fc(last_timestep_out)
        return out

    def fit(self, X, y, num_epochs, batch_size, criterion, optimizer, scheduler=None):
      self.to(self.device)
      train_data = TensorDataset(X, y)
      train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=False)

      best_loss = np.inf
      best_state_dict = None

      for epoch in range(num_epochs):
          self.train()
          total_loss = 0  # Accumulate loss over batches

          # Iterate over the DataLoader
          for X_batch, y_batch in train_loader:
              # Move data to the specified device
              X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

              # Forward pass
              outputs = self(X_batch)

              # Calculate loss
              loss = criterion(outputs, y_batch)

              # Backward pass and optimization
              optimizer.zero_grad()
              loss.backward()
              optimizer.step()

              # Accumulate batch loss
              total_loss += loss.item()

          # Calculate average loss for the epoch
          avg_loss = total_loss / len(train_loader)

          if avg_loss < best_loss:
              best_loss = avg_loss
              best_state_dict = self.state_dict()

          if scheduler is not None:
              scheduler.step(avg_loss)

          # Print loss at intervals
          print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}')

      if best_state_dict is not None:
          self.load_state_dict(best_state_dict)
          print(f"Returned best model with average loss: {best_loss}")

    def predict(self, X, batch_size=1024):
        data_loader = DataLoader(X, batch_size=batch_size, shuffle=False)
        predictions = []
        self.to(self.device)

        with torch.no_grad():
          # Iterate over the DataLoader
          for X_batch in data_loader:
              output_batch = self(X_batch.to(self.device))
              predictions.append(output_batch.cpu().numpy())

        return np.concatenate(predictions, axis=0)