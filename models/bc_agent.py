import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os

class BCNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(BCNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim),
            nn.Softmax(dim=1) # Output probabilities for actions
        )
        
    def forward(self, x):
        return self.net(x)

class BCAgent:
    """
    Behavior Cloning Agent.
    Imitates an 'Expert' (e.g., a profitable rule-based strategy or historical optimal moves).
    """
    def __init__(self, input_dim=8, output_dim=3, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = BCNetwork(input_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()
        
    def train(self, expert_obs, expert_actions, epochs=50, batch_size=32):
        """
        Train the model to mimic expert actions.
        expert_obs: Numpy array of observations (N, input_dim)
        expert_actions: Numpy array of actions (N,) - Class indices (0, 1, 2)
        """
        self.model.train()
        
        # Convert to tensors
        obs_tensor = torch.FloatTensor(expert_obs).to(self.device)
        action_tensor = torch.LongTensor(expert_actions).to(self.device)
        
        dataset = torch.utils.data.TensorDataset(obs_tensor, action_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        print(f"Starting BC Training on {len(expert_obs)} samples...")
        
        for epoch in range(epochs):
            total_loss = 0
            for batch_obs, batch_act in loader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_obs)
                loss = self.criterion(outputs, batch_act)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(loader):.4f}")
                
    def predict(self, obs):
        """
        Predict action for a single observation or batch.
        Returns: Action index (int)
        """
        self.model.eval()
        with torch.no_grad():
            if isinstance(obs, np.ndarray):
                obs = torch.FloatTensor(obs).to(self.device)
            
            # Handle single observation (1D array)
            if obs.dim() == 1:
                obs = obs.unsqueeze(0)
                
            probs = self.model(obs)
            action = torch.argmax(probs, dim=1).item()
            return action
            
    def save(self, path):
        torch.save(self.model.state_dict(), path)
        
    def load(self, path):
        if os.path.exists(path):
            self.model.load_state_dict(torch.load(path))
            self.model.eval()
        else:
            print(f"Warning: BC Model not found at {path}")
