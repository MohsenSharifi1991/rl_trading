import gymnasium as gym
import numpy as np
from gymnasium import spaces
from reward import calculate_reward


class TradingEnv(gym.Env):
    """A simple trading environment for reinforcement learning."""
    def __init__(self, df):
        super(TradingEnv, self).__init__()

        self.info = {}
        self.df = df
        self.max_steps = len(self.df)
        self.observation_space_n = self.df.shape[1]
        self.current_step = 0

        self.action_space = spaces.Discrete(3)  # -1 (short), 0 (hold), 1 (long)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_space_n,), dtype=np.float32)

        self.state = self._next_observation()
        self.done = False

        self.action_t_minus_1 = 0  # Action at time t-1
        self.action_t_minus_2 = 0  # Action at time t-2

        # self.previous_action = 1  # Adjusted to match the action mapping (-1, 0, 1)
        self.previous_price = self.df.iloc[self.current_step]['Close']

        # Parameters for reward calculation - example values
        self.sigma_tgt = 0.1
        self.sigma_t_minus_1 = 0.1
        self.sigma_t_minus_2 = 0.1  # Placeholder, assuming a default or initial value
        self.bp = 20  # Transaction cost in basis points

    def _next_observation(self):
        """Get the next observation from the dataframe."""
        frame = self.df.iloc[self.current_step]
        date = self.df.index[self.current_step]
        # obs = np.array([frame['Close'], frame['macd']])
        obs = frame.values
        # Include the date in the info dictionary for access during the step
        self.info['current_date'] = str(date)  # Convert date to string for simplicity; adjust format as needed
        return obs

    def step(self, action):
        """Take an action and return the next observation, reward, done, info."""
        action_mapped = action - 1  # Map action from {0, 1, 2} to {-1, 0, 1}

        # Check if we are at the end or beyond the last data point
        self.current_step += 1
        if self.current_step >= self.max_steps:
            self.done = True

        # Fetch the current price; use the previous price if we're done
        current_price = self.df.iloc[self.current_step]['Close'] if not self.done else self.previous_price
        # Attempt to fetch current volatility; use the previous period's volatility if we're done
        current_volatility = self.df.iloc[self.current_step]['volatility_60d'] if not self.done else self.sigma_t_minus_1

        # Calculate the price change and the action change
        delta_price_t = current_price - self.previous_price

        # Update reward calculation with the current volatility and action changes
        self.reward = calculate_reward(
            action_t_minus_1=self.action_t_minus_1,
            action_t_minus_2=self.action_t_minus_2,
            delta_price_t= delta_price_t,
            sigma_t=current_volatility,
            sigma_t_minus_1=self.sigma_t_minus_1,
            sigma_t_minus_2=self.sigma_t_minus_2,
            sigma_tgt=self.sigma_tgt,
            bp=self.bp,
            price_t_minus_1=self.previous_price
        )

        # Update actions and volatility history for the next step
        self.action_t_minus_2 = self.action_t_minus_1
        self.action_t_minus_1 = action_mapped

        self.sigma_t_minus_2 = self.sigma_t_minus_1
        self.sigma_t_minus_1 = current_volatility

        self.previous_price = current_price

        # Prepare the next state and information for return
        self.state = self._next_observation() if not self.done else np.zeros(self.observation_space.shape)

        # Collect additional info as needed
        self.info['daily_return'] = delta_price_t / self.previous_price if self.previous_price != 0 else 0

        return self.state, self.reward, self.done, self.info

    def reset(self):
        """Reset the state of the environment to an initial state."""
        self.info = {}
        self.current_step = 0
        self.done = False
        self.action_t_minus_1 = 0
        self.action_t_minus_2 = 0
        self.sigma_t_minus_1 = 0.1
        self.sigma_t_minus_2 = 0.1
        self.previous_price = self.df.iloc[self.current_step]['Close']
        self.state = self._next_observation()
        return self.state
