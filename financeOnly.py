import os
import numpy as np
import csv
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import math
import time
import tensorflow as tf
from tensorflow.keras.layers import GRU, LSTM, Bidirectional, Dense, Flatten, Conv1D, BatchNormalization, LeakyReLU, Dropout
from tensorflow.keras import Sequential
from tensorflow.keras.utils import plot_model
from pickle import load
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import statsmodels.api as sm
from math import sqrt
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
from pickle import dump

import warnings
warnings.filterwarnings("ignore")

from matplotlib import style
plt.style.use('seaborn')

all_stocks = pd.read_csv('stock_yfinance_data.csv')

stock_df = all_stocks[all_stocks['Stock Name'] == "TSLA"] # Select data for TESLA only
stock_df['Date'] = pd.to_datetime(stock_df['Date']) # Format the Date column to pandas datetime objects
stock_df['Date'] = stock_df['Date'].dt.date # Show date only

finance_df = stock_df.copy()
finance_df.iloc[:, 1:] = pd.concat([finance_df.iloc[:, 1:].ffill()])
datetime_series = pd.to_datetime(finance_df['Date'])  # Convert Date column to a pandas datetime series
datetime_index = pd.DatetimeIndex(datetime_series.values) # Creates a new DatetimeIndex objects => Set index to the DateFrame
dataset = finance_df.set_index(datetime_index)  # Sets the datetime_index as the new index of finance_df
dataset = dataset.sort_values(by='Date')
dataset = dataset.drop(columns='Date')

dataset2 = dataset['Close']

def split_train_test(data, test_days=50):
    # test data = latest 20 days
    train_size = len(data) - test_days
    data_train = data[0:train_size]
    data_test = data[train_size:]
    return data_train, data_test

dataset_train, dataset_test = split_train_test(dataset2)
dataset_train, dataset_test = np.array(dataset_train).reshape(-1, 1), np.array(dataset_test).reshape(-1, 1)

scaler = MinMaxScaler(feature_range=(0,1))
# Normalizing values between 0 and 1 using training data to fit
scaler.fit(dataset_train)
scaled_train = scaler.fit_transform(dataset_train)
scaled_test = scaler.fit_transform(dataset_test)

X_train = scaled_train[:-1]
y_train = scaled_train[1:]

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
# Assuming X_train, y_train, X_test, y_test are already loaded as numpy arrays

# Convert numpy arrays to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)

# Create Tensor datasets
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)

# Create Data Loaders
batch_size = 16
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

class AdvancedLSTMModel(nn.Module):
    def __init__(self, input_dim, feature_size, hidden_dim, output_dim, num_layers=3, dropout=0.3
    , device=device):
        super(AdvancedLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size=feature_size,
                            hidden_size=hidden_dim,
                            num_layers=num_layers,
                            batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)  # Dropout only between LSTM layers if num_layers > 1
        self.linear = nn.Linear(hidden_dim, output_dim)  # Output layer
        self.device = device

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.linear(out)
        return out

# Model configuration
input_dim = X_train.shape[0]
feature_size = X_train.shape[1]
output_dim = y_train.shape[1]
hidden_dim = 64

model = AdvancedLSTMModel(input_dim, feature_size, hidden_dim, output_dim)
model = model.to(device)

criterion = nn.MSELoss(reduction='mean')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00)

def train_and_evaluate(model, train_loader, criterion, optimizer, num_epochs=50):
    model.train()
    train_hist = []
    test_hist = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_train_loss = 0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            predictions = model(batch_X)
            loss = criterion(predictions, batch_y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        avg_train_loss = total_train_loss / len(train_loader)
        train_hist.append(avg_train_loss)

        print(f'Epoch {epoch+1}, Train Loss: {avg_train_loss:.4f}')

    return train_hist

train_losses = train_and_evaluate(model, train_loader, criterion, optimizer)
num_forecast_steps = 50 #20

historical_data = X_test[0]  # This is the last known sequence to start predictions from
forecasted_values = []

model.eval()
model.to(device)
with torch.no_grad():
    for _ in range(num_forecast_steps):
        # 1 tensor data in a batch
        historical_data_tensor = torch.tensor(historical_data, dtype=torch.float32).unsqueeze(0).to(device)
        #print(historical_data_tensor.shape)

        predicted_tensor = model(historical_data_tensor)
        #print(predicted_tensor.shape)

        predicted_value = predicted_tensor.cpu().numpy()[0,0]
        #print(predicted_value)
        forecasted_values.append(predicted_value.tolist())
        historical_data = np.roll(historical_data, shift=-1)
        historical_data[-1] = predicted_value  # Update the sequence with the new prediction
        
dates = dataset2.index[-50:]
pred_price = scaler.inverse_transform(forecasted_values)[:,0]

fig, ax = plt.subplots(figsize=(11,6))
ax.plot(dataset2.index[-100:], dataset2[-100:], label='Actual Values', color='#008B8B', alpha=0.25)
ax.plot(dates, pred_price, label='Predicted Prices', color='#008B8B', linestyle='--', alpha=1)

ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('USD', fontsize=14, color='#008B8B')
ax.tick_params(axis='y', labelcolor='#008B8B')

ax.set_title(f"Tesla: Stock Price Prediction", fontsize=16)

lines, labels = ax.get_legend_handles_labels()
ax.legend(lines, labels) #loc='upper right'


plt.xlim(dataset2.index[-100], dataset2.index[-1])
ax.grid(False)
plt.show()