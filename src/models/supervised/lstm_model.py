import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
from typing import Dict

class LSTMNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :]) # Take last time step
        return self.softmax(out)

class LSTMModel:
    def __init__(self, model_path: str = "models/lstm_direction.pth"):
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.sequence_length = 60 # 1 hour lookback
        self.input_dim = 5 # OHLCV (normalized)
        
    def prepare_data(self, df: pd.DataFrame):
        # Normalize
        df_norm = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_norm[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
        
        df_norm = df_norm.dropna()
        data = df_norm[['open', 'high', 'low', 'close', 'volume']].values
        
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            # Target: 1 if next close > current close, else 0
            target = 1 if df['close'].iloc[i + self.sequence_length] > df['close'].iloc[i + self.sequence_length - 1] else 0
            y.append(target)
            
        return np.array(X), np.array(y)

    def train(self, df: pd.DataFrame, epochs=10):
        X, y = self.prepare_data(df)
        
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.LongTensor(y).to(self.device)
        
        self.model = LSTMNet(self.input_dim, 64, 2, 2).to(self.device)
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 1 == 0:
                print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
                
        self.save()

    def predict(self, df: pd.DataFrame) -> Dict[str, float]:
        if self.model is None:
            self.load()
            
        # Prepare single sequence
        if len(df) < self.sequence_length + 60: # Need extra for rolling norm
            return {'down': 0.5, 'up': 0.5}
            
        df_norm = df.copy()
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df_norm[col] = (df[col] - df[col].rolling(60).mean()) / df[col].rolling(60).std()
            
        seq = df_norm[['open', 'high', 'low', 'close', 'volume']].iloc[-self.sequence_length:].values
        X_tensor = torch.FloatTensor([seq]).to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = torch.exp(self.model(X_tensor)) # Convert LogSoftmax to Prob
            probs = outputs.cpu().numpy()[0]
            
        return {'down': float(probs[0]), 'up': float(probs[1])}

    def save(self):
        if not os.path.exists(os.path.dirname(self.model_path)):
            os.makedirs(os.path.dirname(self.model_path))
        torch.save(self.model.state_dict(), self.model_path)

    def load(self):
        self.model = LSTMNet(self.input_dim, 64, 2, 2).to(self.device)
        if os.path.exists(self.model_path):
            self.model.load_state_dict(torch.load(self.model_path))
            self.model.eval()
        else:
            # Initialize random for dev
            pass
