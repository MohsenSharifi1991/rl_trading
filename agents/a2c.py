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


class MLPValue(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super(MLPValue, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class A2C:
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
        self.value_net = MLPValue(state_dim, hidden_dim=cfg.hidden_dim).to(self.device)

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=cfg.lr) # optimizer
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=cfg.lr)  # optimizer

        self.update_freq = 10
        self.update_counter = 0  # Initialize a counter to control the verbosity of the print statement

    def choose_action(self, state):
        state = torch.tensor([state], device=self.device, dtype=torch.float32)
        action_probs = self.policy_net(state)
        distribution = torch.distributions.Categorical(action_probs)
        action = distribution.sample()
        log_prob = distribution.log_prob(action)
        return action.item(), log_prob

    # def compute_discounted_rewards(self, rewards):
    #     """Compute and normalize discounted rewards."""
    #     Gt, discounted_rewards = 0, []
    #     for reward in reversed(rewards):
    #         Gt = reward + self.gamma * Gt
    #         discounted_rewards.insert(0, Gt)
    #     discounted_rewards = torch.tensor(discounted_rewards, dtype=torch.float32, device=self.device)
    #     return (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
    #
    # def update_policy(self, log_probs, advantages):
    #     """Calculate and apply policy gradient."""
    #     policy_loss = []
    #     for log_prob, advantage in zip(log_probs, advantages):
    #         policy_loss.append(-log_prob * advantage)  # Gradient ascent, hence the negative sign
    #
    #     policy_loss = torch.stack(policy_loss).sum()
    #
    #     self.optimizer.zero_grad()
    #     policy_loss.backward()
    #     self.optimizer.step()
    #
    # def update_value_network(self, saved_states, discounted_rewards):
    #     """Update the value network."""
    #     saved_states_tensor = torch.cat(saved_states).to(self.device)
    #     state_values = self.value_net(saved_states_tensor).squeeze()
    #
    #     value_loss = F.mse_loss(state_values, discounted_rewards)
    #
    #     self.value_optimizer.zero_grad()
    #     value_loss.backward()
    #     self.value_optimizer.step()

    # def update(self, rewards, log_probs, saved_states):
    #     """Perform updates for both policy and value networks."""
    #     discounted_rewards = self.compute_discounted_rewards(rewards)
    #
    #     # Convert saved states to a tensor for batch processing
    #     saved_states_tensor = torch.cat(saved_states).to(self.device)
    #     with torch.no_grad():
    #         state_values = self.value_net(saved_states_tensor).squeeze()
    #
    #     advantages = discounted_rewards - state_values.detach()
    #
    #     # Update networks
    #     self.update_policy(log_probs, advantages)
    #     self.update_value_network(saved_states, discounted_rewards)



    def update(self, rewards, log_probs, saved_states, next_states):
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)

        # Step 2: Prepare state values for advantage calculation.
        # Convert saved states into a tensor for batch processing through the value network.
        saved_states_tensor = torch.cat(saved_states).to(self.device)
        # Pass state tensor through the value network to estimate state values.
        state_values = self.value_net(saved_states_tensor).squeeze()

        next_states_tensor = torch.cat(next_states).to(self.device)
        next_state_values = self.value_net(next_states_tensor).squeeze()

        # Step 3: Compute TD errors for advantage calculation
        # TD Error = reward + gamma * next_state_value - state_value
        td_errors = rewards + self.gamma * next_state_values.detach() - state_values.detach()

        # Step 4: Calculate policy loss using log probabilities and advantages.
        policy_gradient = []
        for log_prob, advantage in zip(log_probs, td_errors):
            policy_gradient.append(-log_prob * advantage)  # Gradient ascent, hence the negative sign
        policy_gradient = torch.stack(policy_gradient).sum()


        # Update the policy network
        self.optimizer.zero_grad()
        policy_gradient.backward()
        self.optimizer.step()

        # Step 6: Update the value network.
        # No need to recompute state values here; use the ones already computed for efficiency.
        # Compute value loss as the mean squared error between predicted state values and discounted rewards.
        value_loss = F.mse_loss(state_values, rewards + self.gamma * next_state_values)
        self.value_optimizer.zero_grad()  # Clear gradients for value network.
        value_loss.backward()  # Compute gradients for value network parameters.
        self.value_optimizer.step()  # Update value network parameters.

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path + 'a2c_checkpoint.pth')

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path + 'a2c_checkpoint.pth'))
