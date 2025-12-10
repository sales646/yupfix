"""
PPO Trainer for Position Sizing
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import logging

logger = logging.getLogger("PPOTrainer")

class PPOTrainer:
    def __init__(self, env, policy, config):
        self.env = env
        self.policy = policy
        self.config = config
        
        self.optimizer = optim.Adam(
            self.policy.parameters(), 
            lr=config.get('lr', 3e-4)
        )
        
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.value_coef = config.get('value_coef', 0.5)
        self.entropy_coef = config.get('entropy_coef', 0.01)
        self.max_grad_norm = config.get('max_grad_norm', 0.5)
        self.ppo_epochs = config.get('ppo_epochs', 10)
        self.batch_size = config.get('batch_size', 64)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        
    def collect_rollouts(self, steps=2048):
        """
        Collect trajectories from the environment.
        """
        obs_buf, act_buf, rew_buf, val_buf, logp_buf = [], [], [], [], []
        
        obs, _ = self.env.reset()
        done = False
        
        episode_rewards = []
        cur_ep_ret = 0
        
        for step in range(steps):
            # Prepare observation
            # Obs shape from env might be (features,) -> need (1, seq_len, features) for Mamba
            # This requires the Env to return a sequence history buffer
            
            # For now, assume obs is already correct shape or we reshape it
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device) # (1, features)
            # Mamba expects (B, L, D). If env returns single step, we might need a history buffer wrapper.
            # Let's assume the Env returns a window (L, D)
            
            if obs_tensor.dim() == 2: # (1, features) -> (1, 1, features)
                 obs_tensor = obs_tensor.unsqueeze(1)
            
            with torch.no_grad():
                action, log_prob, value = self.policy.get_action(obs_tensor)
                
            action_np = action.cpu().numpy()[0]
            val_np = value.item()
            log_prob_np = log_prob.item()
            
            # Step env
            next_obs, reward, terminated, truncated, _ = self.env.step(action_np)
            done = terminated or truncated
            
            # Store
            obs_buf.append(obs_tensor.cpu()) # Store as tensor to save conversion later
            act_buf.append(torch.FloatTensor(action).cpu())
            rew_buf.append(reward)
            val_buf.append(val_np)
            logp_buf.append(log_prob_np)
            
            obs = next_obs
            cur_ep_ret += reward
            
            if done:
                episode_rewards.append(cur_ep_ret)
                cur_ep_ret = 0
                obs, _ = self.env.reset()
                
        # Bootstrap value if not done
        if not done:
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if obs_tensor.dim() == 2: obs_tensor = obs_tensor.unsqueeze(1)
            with torch.no_grad():
                _, last_val = self.policy(obs_tensor)
                last_val = last_val.item()
        else:
            last_val = 0
            
        # Compute GAE
        rews = np.array(rew_buf + [last_val])
        vals = np.array(val_buf + [last_val])
        
        adv_buf = np.zeros_like(rew_buf, dtype=np.float32)
        last_gae_lam = 0
        
        for t in reversed(range(len(rew_buf))):
            delta = rews[t] + self.gamma * vals[t+1] - vals[t]
            adv_buf[t] = last_gae_lam = delta + self.gamma * self.gae_lambda * last_gae_lam
            
        ret_buf = adv_buf + vals[:-1]
        
        return {
            'obs': torch.cat(obs_buf),
            'act': torch.cat(act_buf),
            'ret': torch.FloatTensor(ret_buf),
            'adv': torch.FloatTensor(adv_buf),
            'logp': torch.FloatTensor(logp_buf)
        }, np.mean(episode_rewards) if episode_rewards else 0
        
    def update(self, rollouts):
        """
        Update policy using PPO.
        """
        obs = rollouts['obs'].to(self.device)
        act = rollouts['act'].to(self.device)
        ret = rollouts['ret'].to(self.device)
        adv = rollouts['adv'].to(self.device)
        old_logp = rollouts['logp'].to(self.device)
        
        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        
        dataset_size = obs.size(0)
        indices = np.arange(dataset_size)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, dataset_size, self.batch_size):
                end = start + self.batch_size
                idx = indices[start:end]
                
                batch_obs = obs[idx]
                batch_act = act[idx]
                batch_ret = ret[idx]
                batch_adv = adv[idx]
                batch_old_logp = old_logp[idx]
                
                # Evaluate current policy
                dist, values = self.policy(batch_obs)
                log_probs = dist.log_prob(batch_act).sum(-1)
                entropy = dist.entropy().mean()
                
                # Ratio
                ratio = torch.exp(log_probs - batch_old_logp)
                
                # Surrogate Loss
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value Loss
                value_loss = 0.5 * ((values.squeeze() - batch_ret) ** 2).mean()
                
                # Total Loss
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
        return loss.item()

    def train(self, total_timesteps):
        steps = 0
        while steps < total_timesteps:
            rollouts, avg_reward = self.collect_rollouts(steps=2048)
            loss = self.update(rollouts)
            steps += 2048
            
            logger.info(f"Steps: {steps}, Avg Reward: {avg_reward:.2f}, Loss: {loss:.4f}")
