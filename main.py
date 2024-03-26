from agents.a2c import A2C
from agents.dqn import DQN
from agents.pg import PG, PGContinuous
from agents.pgbaseline import PGBaseline
from config import Config
from evaluation import evaluate_strategy
from rl_trainer import RLTrainer
from trading_env import TradingEnv
from utils.load_data import load_and_process_data, split_data
import os
import matplotlib.pyplot as plt
from test import test
from utils.plots import plot_cumulative_trade_return

if __name__ == '__main__':
    cfg = Config()
    tickers = {
        "AUD/USD": "FXA",
        "GBP/USD": "GBPUSD=X",
        "USD/CAD": "CAD=X",
        "US Dollar Index": "UUP",
        "EUR/USD": "FXE",
        "JPY/USD": "FXY",
        "MXN/USD": "MXNUSD=X",
        "Nikkei 225": "EWJ",
        "CHF/USD": "CHFUSD=X",
    }
    tickers = {"EUR/USD": "FXE"}
    features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
    # Load and process the data
    hist_data = load_and_process_data(tickers, features, combine=False)

    # Split data into training and test sets
    train_data, test_data = split_data(hist_data, split_date="2016-01-01")
    train_data_df, test_data_df = train_data['EUR/USD'], test_data['EUR/USD']
    train_data_df, test_data_df = train_data_df.drop(['Currency_Pair'], axis=1), test_data_df.drop(['Currency_Pair'], axis=1)

    # Assuming we are using the discrete trading environment for this example
    env = TradingEnv(train_data_df)
    state_space_dim = env.observation_space.shape[0]
    action_space_dim = env.action_space.n

    # Example with PG agent, adjust as necessary for other agents like DQN, PGBaseline, A2C, etc.
    if cfg.algo_name == 'DQN':
        agent = DQN(state_space_dim, action_space_dim, cfg)
    elif cfg.algo_name == 'PG':
        agent = PG(state_space_dim, action_space_dim, cfg)
    elif cfg.algo_name == 'PGBaseline':
        agent = PGBaseline(state_space_dim, action_space_dim, cfg)
    elif cfg.algo_name == 'A2C':
        agent = A2C(state_space_dim, action_space_dim, cfg)
    else:
        raise NotImplementedError(f"Agent for {cfg.algo_name} not implemented.")

    trainer = RLTrainer(cfg, env, agent)
    # Train the agent
    rewards, ma_rewards = trainer.train()

    os.makedirs(cfg.result_path, exist_ok=True)  # create output folders if they don't exist
    os.makedirs(cfg.model_path, exist_ok=True)
    agent.save(path=cfg.model_path)  # save model

    # Plot the training results
    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    ax.plot(list(range(1, cfg.train_eps + 1)), rewards, color='blue', label='rewards')
    ax.plot(list(range(1, cfg.train_eps + 1)), ma_rewards, color='green', label='ma_rewards')
    ax.legend()
    ax.set_xlabel('Episode')
    plt.savefig(cfg.result_path + 'train.jpg')

    # testing
    # Initialize the test environment with test data
    test_env = TradingEnv(test_data_df)

    # Load the trained model
    agent.load(path=cfg.model_path)

    # Evaluate the agent on the test dataset
    average_reward, daily_returns = test(cfg, test_env, agent)

    metrics = evaluate_strategy(daily_returns)
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    daily_returns.plot()
    plt.show()
    # Plot the cumulative trade return
    plot_cumulative_trade_return(daily_returns)
    # plt.savefig(cfg.result_path + 'test.jpg')







# Pseudocode for DRL model implementation

# # Load and preprocess data
# data = load_data('path_to_data')
# processed_data = preprocess_data(data)
#
# # Define the trading environment
# environment = TradingEnvironment(data=processed_data)
#
# # Initialize the DRL models
# dqn_model = DQN(environment.state_size, environment.action_size)
# pg_model = PG(environment.state_size, environment.action_size)
# a2c_model = A2C(environment.state_size, environment.action_size)
#
# # Train models
# dqn_model.train(environment)
# pg_model.train(environment)
# a2c_model.train(environment)
#
# # Evaluate models
# dqn_performance = evaluate_model(dqn_model, environment)
# pg_performance = evaluate_model(pg_model, environment)
# a2c_performance = evaluate_model(a2c_model, environment)
#
# # Compare with baseline strategies
# compare_performance(dqn_performance, pg_performance, a2c_performance, baseline_strategies)