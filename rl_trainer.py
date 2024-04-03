import torch
from tqdm import tqdm
# TODO: use memory for all training methods instead of calling each. to make it consistant, reinfroce has epoisodic memory while the dqn more about reaching to limit
# TODO update the each agent to become consistant format

class RLTrainer:
    def __init__(self, cfg, env, agent):
        self.cfg = cfg
        self.env = env
        self.agent = agent
        self.rewards = []
        self.ma_rewards = []

    def train(self):
        '''Dispatcher to call appropriate training method based on the algorithm name.'''
        if self.cfg.algo_name == 'DQN':
            return self.train_dqn()
        elif self.cfg.algo_name == 'PG':
            return self.train_reinforce()
        elif self.cfg.algo_name == 'PGBaseline':
            return self.train_reinforce_with_baseline()
        elif self.cfg.algo_name == 'A2C':
            return self.train_a2c()
        else:
            raise NotImplementedError(f"Training method for {self.cfg.algo_name} not implemented.")

    def train_dqn(self):
        '''Training using the DQN algorithm.'''
        print('Start Training DQN!')
        print(f'Environment: {self.cfg.env_name}, Algorithm: {self.cfg.algo_name}, Device: {self.cfg.device}')

        for i_ep in tqdm(range(self.cfg.train_eps), desc="Training Progress"):
            ep_reward = 0
            state = self.env.reset()
            while True:
                action, _ = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.agent.memory.push(state, action, reward, next_state, done)  # save transition
                state = next_state
                self.agent.update()
                ep_reward += reward
                if done:
                    break

            if (i_ep + 1) % self.cfg.target_update == 0:  # update target network
                self.agent.target_net.load_state_dict(self.agent.policy_net.state_dict())

            self.rewards.append(ep_reward)
            self._update_ma_rewards(ep_reward)
            self._print_progress(i_ep, ep_reward)

        print('Finish Training!')
        return self.rewards, self.ma_rewards

    def train_reinforce(self):
        '''Training using the REINFORCE algorithm.'''
        print('Start Training REINFORCE!')
        print(f'Environment: {self.cfg.env_name}, Algorithm: {self.cfg.algo_name}, Device: {self.cfg.device}')

        for i_ep in tqdm(range(self.cfg.train_eps), desc="Training Progress"):
            ep_reward = 0
            state = self.env.reset()
            saved_log_probs = []  # Store log probabilities of the actions taken
            ep_rewards = []  # Store rewards for each step in the episode

            while True:
                action, log_prob = self.agent.choose_action(state)  # Assume agent returns log_prob too
                next_state, reward, done, _ = self.env.step(action)

                saved_log_probs.append(log_prob)  # Save the log probability
                ep_rewards.append(reward)  # Save the reward

                state = next_state
                ep_reward += reward

                if done:
                    break

            self.agent.update(ep_rewards, saved_log_probs)  # Update policy once the episode is done

            self.rewards.append(ep_reward)
            self._update_ma_rewards(ep_reward)
            # self._print_progress(i_ep, ep_reward)

        print('Finish Training REINFORCE!')
        return self.rewards, self.ma_rewards

    def train_reinforce_with_baseline(self):
        '''Training using the REINFORCE with baseline algorithm.'''
        print('Start Training REINFORCE with Baseline!')
        print(f'Environment: {self.cfg.env_name}, Algorithm: {self.cfg.algo_name}, Device: {self.cfg.device}')

        for i_ep in tqdm(range(self.cfg.train_eps), desc="Training Progress"):
            ep_reward = 0
            state = self.env.reset()
            saved_states = []  # To store states for baseline
            saved_log_probs = []  # Store log probabilities of the actions taken
            ep_rewards = []  # Store rewards for each step in the episode

            while True:
                action, log_prob = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                # Ensure state is a tensor when appended
                saved_states.append(torch.tensor([state], device=self.cfg.device, dtype=torch.float32))

                saved_log_probs.append(log_prob)  # Save the log probability
                ep_rewards.append(reward)  # Save the reward

                state = next_state
                ep_reward += reward

                if done:
                    break

            self.agent.update(ep_rewards, saved_log_probs, saved_states)  # Update policy once the episode is done

            self.rewards.append(ep_reward)
            self._update_ma_rewards(ep_reward)
            self._print_progress(i_ep, ep_reward)

        print('Finish Training REINFORCE with Baseline!')
        return self.rewards, self.ma_rewards

    def train_a2c(self):
        '''Training using the A2C algorithm.'''
        print('Start Training A2C!')
        print(f'Environment: {self.cfg.env_name}, Algorithm: {self.cfg.algo_name}, Device: {self.cfg.device}')

        for i_ep in tqdm(range(self.cfg.train_eps), desc="Training Progress"):
            ep_reward = 0
            state = self.env.reset()
            saved_states = []  # To store states for value function updates
            saved_next_states = []  # Store next states for TD error calculation
            saved_log_probs = []  # Store log probabilities of the actions taken
            saved_rewards = []  # Store rewards for each step in the episode
            saved_dones = []  # Store done flags for each step in the episode

            while True:
                action, log_prob = self.agent.choose_action(state)
                next_state, reward, done, _ = self.env.step(action)

                saved_states.append(torch.tensor([state], device=self.cfg.device, dtype=torch.float32))
                saved_next_states.append(torch.tensor([next_state], device=self.cfg.device, dtype=torch.float32))
                saved_log_probs.append(log_prob)
                saved_rewards.append(reward)
                saved_dones.append(done)

                state = next_state
                ep_reward += reward

                if done:
                    self.agent.update(saved_states, saved_rewards, saved_log_probs, saved_next_states, saved_dones)
                    break

            self.rewards.append(ep_reward)
            self._update_ma_rewards(ep_reward)
            self._print_progress(i_ep, ep_reward)

        print('Finish Training A2C!')
        return self.rewards, self.ma_rewards

    def _update_ma_rewards(self, ep_reward):
        '''Update moving average of rewards.'''
        if self.ma_rewards:
            self.ma_rewards.append(0.9 * self.ma_rewards[-1] + 0.1 * ep_reward)
        else:
            self.ma_rewards.append(ep_reward)

    def _print_progress(self, i_ep, ep_reward):
        '''Print progress every few episodes.'''
        if (i_ep + 1) % 10 == 0:
            print(f'Episode: {i_ep + 1}/{self.cfg.train_eps}, Reward: {ep_reward}')
