import pandas as pd
from tqdm import tqdm

def test(cfg, env, agent):
    print('Start Testing!')
    print(f'Environment：{cfg.env_name}, Algorithm：{cfg.algo_name}, Device：{cfg.device}')
    ############# Test does not use e-greedy policy, so we set epsilon to 0 ###############
    cfg.epsilon_start = 0.0
    cfg.epsilon_end = 0.0
    cfg.num_episodes = 1
    ################################################################################
    rewards = []  # record total rewards
    all_dates = []  # List to store all dates across episodes (if num_episodes > 1)
    all_daily_returns = []  # List to store all daily returns across episodes
    all_daily_rewards = []
    for i_ep in range(cfg.num_episodes):
        ep_reward = 0
        ep_daily_returns = []
        state = env.reset()
        while True:
            action, _ = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            all_dates.append(info['current_date'])  # Store date
            all_daily_returns.append(info['daily_return'])  # Store daily retur
            all_daily_rewards.append(reward)  # Store daily reward
            if done:
                break
        rewards.append(ep_reward)

        print(f"Episode：{i_ep + 1}，Reward：{ep_reward:.1f}")
    average_reward = sum(rewards) / len(rewards)
    print(f"Average Reward over {cfg.num_episodes} episodes: {average_reward}")
    print('Finish Testing!')

    # Convert dates and daily rewards into a pandas DataFrame
    results_df = pd.DataFrame({'Date': all_dates, 'Daily Reward': all_daily_rewards})
    results_df['Date'] = pd.to_datetime(results_df['Date'])  # Ensure the 'Date' column is in datetime format
    results_df.set_index('Date', inplace=True)  # Set the 'Date' column as the index
    return average_reward, results_df


def run_baseline(cfg, env, agent):
    print('Start Testing!')
    # print(f'Environment: {cfg.env_name}, Algorithm: {agent.cfg.algo_name}, Device: {cfg.device}')

    # Initialize rewards tracking
    rewards = []
    ma_rewards = []  # Moving Average Rewards
    all_dates = []  # List to store all dates across episodes
    all_daily_rewards = []  # List to store all daily rewards across episodes

    for i_ep in tqdm(range(cfg.baseline_eps), desc="Testing Progress"):
        ep_reward = 0
        state = env.reset()
        while True:
            action, _ = agent.choose_action(state)
            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            if 'current_date' in info:
                all_dates.append(info['current_date'])  # Store date
            all_daily_rewards.append(reward)  # Store daily reward
            if done:
                break

        # Store episode reward
        rewards.append(ep_reward)
        # Update and store moving average of rewards
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)

        print(f"Episode: {i_ep + 1}, Reward: {ep_reward:.1f}")

    average_reward = sum(rewards) / len(rewards)
    print(f"Average Reward over {cfg.baseline_eps} episodes: {average_reward:.2f}")
    print('Finish Testing!')

    # Convert dates and daily rewards into a pandas DataFrame
    results_df = pd.DataFrame({'Date': all_dates, 'Daily Reward': all_daily_rewards})
    results_df['Date'] = pd.to_datetime(results_df['Date'])  # Ensure the 'Date' column is in datetime format
    results_df.set_index('Date', inplace=True)  # Set the 'Date' column as the index

    return rewards, ma_rewards, results_df