import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import math
import numpy as np


class MLP(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=-1)  # Use softmax to get action probabilities

class MLPContinuous(nn.Module):
    def __init__(self, state_dim, hidden_dim=128, action_dim=1):
        super(MLPContinuous, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))  # Log std as a learnable parameter

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        mean = self.mean(x)
        std = torch.exp(self.log_std)  # Ensure std is positive
        return mean, std

class PG:
    """
    REINFORCE: Monte Carlo Policy Gradient
    """
    def __init__(self, state_dim, action_dim, cfg):
        self.action_dim = action_dim
        self.device = cfg.device  # cpu or gpu
        self.gamma = cfg.gamma  # discount factor
        self.frame_idx = 0  # attenuation

        self.batch_size = cfg.batch_size
        self.policy_net = MLP(state_dim, action_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # optimizer

        self.update_freq = 10
        self.update_counter = 0  # Initialize a counter to control the verbosity of the print statement

    def choose_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        action_probs = self.policy_net(state)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob

    def compute_discounted_rewards(self, rewards):
        """Compute and normalize discounted rewards."""
        Gt, discounted_rewards = 0, []
        for reward in reversed(rewards):
            Gt = reward + self.gamma * Gt
            discounted_rewards.insert(0, Gt)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
        return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)


    def update_policy(self, log_probs, discounted_rewards):
        """Calculate and apply policy gradient."""
        policy_loss = []
        for log_prob, advantage in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * advantage)  # Gradient ascent, hence the negative sign

        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def update(self, rewards, log_probs):
        discounted_rewards = self.compute_discounted_rewards(rewards)
        self.update_policy(log_probs, discounted_rewards)


    # def update(self, rewards, log_probs):
    #     Gt, discounted_rewards = 0, []
    #     for reward in reversed(rewards):
    #         Gt = reward + self.gamma * Gt
    #         discounted_rewards.insert(0, Gt)
    #
    #     discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
    #     discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
    #                 discounted_rewards.std() + 1e-9)  # Normalize
    #
    #     policy_gradient = []
    #     for log_prob, Gt in zip(log_probs, discounted_rewards):
    #         policy_gradient.append(-log_prob * Gt)
    #
    #     self.optimizer.zero_grad()
    #     policy_gradient = torch.stack(policy_gradient).sum()
    #     policy_gradient.backward()
    #     self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_checkpoint.pth'))


class PGContinuous:  # Inheriting from the original PG for simplicity
    """
    REINFORCE: Monte Carlo Policy Gradient
    """
    def __init__(self, state_dim, cfg):
        self.device = cfg.device  # cpu or gpu
        self.gamma = cfg.gamma  # discount factor
        self.frame_idx = 0  # attenuation

        self.batch_size = cfg.batch_size
        self.policy_net = MLPContinuous(state_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # optimizer

        self.update_freq = 10
        self.update_counter = 0  # Initialize a counter to control the verbosity of the print statement

    def choose_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        mean, std = self.policy_net(state)
        distribution = torch.distributions.Normal(mean, std)
        # distribution = torch.distributions.TruncatedNormal(low=-1.0, high=1.0, mean=mean, std=std) # Ensure action is within [-1, 1]
        action = distribution.sample()
        action_clipped = torch.clamp(action, min=-1.0, max=1.0)
        log_prob = distribution.log_prob(action)
        return action_clipped.item(), log_prob
    # TODO what should we do about clipped action space and log_prob driven from original action

    def compute_discounted_rewards(self, rewards):
        """Compute and normalize discounted rewards."""
        Gt, discounted_rewards = 0, []
        for reward in reversed(rewards):
            Gt = reward + self.gamma * Gt
            discounted_rewards.insert(0, Gt)
        discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
        return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)


    def update_policy(self, log_probs, discounted_rewards):
        """Calculate and apply policy gradient."""
        policy_loss = []
        for log_prob, advantage in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * advantage)  # Gradient ascent, hence the negative sign

        policy_loss = torch.stack(policy_loss).sum()

        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

    def update(self, rewards, log_probs):
        discounted_rewards = self.compute_discounted_rewards(rewards)
        self.update_policy(log_probs, discounted_rewards)

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'pg_cont_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'pg_cont_checkpoint.pth'))

