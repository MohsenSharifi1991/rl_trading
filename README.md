# Deep Reinforcement Learning for Trading

This project applies Deep Reinforcement Learning (DRL) algorithms to develop trading strategies for continuous futures forex markets. By leveraging state-of-the-art reinforcement learning techniques, the project aims to optimize trading decisions directly, thereby potentially outperforming traditional trading strategies.

## Project Overview

The project focuses on utilizing three main DRL algorithms:

- Deep Q-Learning Network (DQN)
- Policy Gradients (PG) and PGBaseline
- Advantage Actor-Critic (A2C)

These algorithms are applied to process and analyze financial data from 2011 to 2019, aiming to generate actionable trading signals that can navigate the complexities of various market conditions.
## Methodology

### State Space

The state space includes historical data features such as normalized price returns, MACD (Moving Average Convergence Divergence), and RSI (Relative Strength Index) indicators. At each timestep, the state captures the current market conditions based on these features.

### Action Space

Two types of action spaces are explored:

- **Discrete Action Space**: Actions are defined as {-1, 0, 1}, representing sell, hold, and buy positions.
- **Continuous Action Space**: Actions can take any value within [-1, 1], allowing for a more nuanced control over the position sizes.

### Reward Function

The reward function is designed to maximize profits, incorporating volatility scaling to adjust trade positions based on market volatility. It factors in transaction costs, encouraging strategies that balance the pursuit of profit with the cost of trading.

## Getting Started

### Installation

1. Clone the repository to your local machine:
```git clone https://github.com/MohsenSharifi1991/rl_trading.git ```

2. Navigate to the cloned repository:
 ```cd rl_trading ```
3. Install the required Python libraries:
 ```pip install -r requirements.txt```

### Dataset

## Usage

To run the trading model training and evaluation, execute the main script:


This script will preprocess the data, train the DRL models on the training dataset, and evaluate their performance on the testing dataset. The results will include metrics such as annualized return, Sharpe ratio, and maximum drawdown, comparing the DRL models against baseline trading strategies.

## Acknowledgments

- https://github.com/lukesalamone/deep-q-trading-agent
- https://medium.com/@murrawang/deep-q-network-and-its-application-in-algorithmic-trading-16440a112e04
- Zhang, Zihao, Stefan Zohren, and Stephen Roberts. "Deep reinforcement learning for trading." arXiv preprint arXiv:1911.10107 (2019).

