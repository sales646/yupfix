"""
PPO Agent for Portfolio Management
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import logging

logger = logging.getLogger("PPOAgent")

class PPOAgent:
    def __init__(self, env, config):
        self.env = env
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Policy Network (Actor)
        # Input: Observation -> Output: Mean, Std for actions
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, act_dim),
            nn.Tanh() # Output [-1, 1]
        ).to(self.device)
        
        self.log_std = nn.Parameter(torch.zeros(act_dim).to(self.device))
        
        # Value Network (Critic)
        self.critic = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        ).to(self.device)
        
        self.optimizer = optim.Adam([
            {'params': self.actor.parameters(), 'lr': 3e-4},
            {'params': self.log_std, 'lr': 3e-4},
            {'params': self.critic.parameters(), 'lr': 1e-3}
        ])
        
    def get_action(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        mean = self.actor(obs)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        
        return action.cpu().numpy(), log_prob.cpu().item(), dist.mean.cpu().detach().numpy()
        
    def compute_gae(self, rewards, values, dones, gamma=0.99, lam=0.95):
        advantages = []
        last_gae = 0
        
        # Append value of next state (0 if done)
        values = values + [0]
        
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * values[t+1] * (1 - dones[t]) - values[t]
            last_gae = delta + gamma * lam * (1 - dones[t]) * last_gae
            advantages.insert(0, last_gae)
            
        return advantages
        
    def update_policy(self, states, actions, old_log_probs, returns, advantages, clip_ratio=0.2):
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(old_log_probs).to(self.device)
        returns = torch.FloatTensor(returns).to(self.device)
        advantages = torch.FloatTensor(advantages).to(self.device)
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # New prob
        mean = self.actor(states)
        std = self.log_std.exp().expand_as(mean)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(-1)
        entropy = dist.entropy().mean()
        
        # Ratio
        ratio = torch.exp(log_probs - old_log_probs)
        
        # Surrogate loss
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Value loss
        values = self.critic(states).squeeze()
        value_loss = 0.5 * ((values - returns) ** 2).mean()
        
        # Total loss
        loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()

    def train(self, total_timesteps=10000):
        obs, _ = self.env.reset()
        
        states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []
        
        for step in range(total_timesteps):
            action, log_prob, _ = self.get_action(obs)
            value = self.critic(torch.FloatTensor(obs).to(self.device)).item()
            
            next_obs, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            values.append(value)
            dones.append(done)
            log_probs.append(log_prob)
            
            obs = next_obs
            
            if done:
                obs, _ = self.env.reset()
                
            # Update every N steps (e.g., 2048)
            if len(states) >= 2048:
                advantages = self.compute_gae(rewards, values, dones)
                returns = [adv + val for adv, val in zip(advantages, values)]
                
                loss = self.update_policy(states, actions, log_probs, returns, advantages)
                logger.info(f"Step {step}: Loss {loss:.4f}")
                
                states, actions, rewards, values, dones, log_probs = [], [], [], [], [], []
