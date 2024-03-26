import numpy as np
from scipy.stats import norm

def evaluate_strategy(daily_returns):
    """
    Evaluates the performance of a trading strategy.

    Parameters:
    - daily_returns: A list or numpy array of daily returns.

    Returns:
    - A dictionary containing performance metrics.
    """
    daily_returns = np.array(daily_returns)
    annualized_return = np.mean(daily_returns) * 252
    annualized_std = np.std(daily_returns) * np.sqrt(252)
    downside_std = np.std(daily_returns[daily_returns < 0]) * np.sqrt(252)

    sharpe_ratio = annualized_return / annualized_std
    sortino_ratio = annualized_return / downside_std

    drawdowns = np.maximum.accumulate(np.cumprod(1 + daily_returns)) - np.cumprod(1 + daily_returns)
    max_drawdown = np.max(drawdowns)

    calmar_ratio = annualized_return / max_drawdown

    percent_positive_returns = np.mean(daily_returns > 0)

    # Assuming the average P&L ratio can be simplified as the mean of positive returns over the mean of negative returns
    average_p_l_ratio = np.mean(daily_returns[daily_returns > 0]) / -np.mean(daily_returns[daily_returns < 0])

    return {
        "E(R)": annualized_return,
        "Std(R)": annualized_std,
        "Downside Deviation": downside_std,
        "Sharpe Ratio": sharpe_ratio,
        "Sortino Ratio": sortino_ratio,
        "Maximum Drawdown": max_drawdown,
        "Calmar Ratio": calmar_ratio,
        "% of Positive Returns": percent_positive_returns,
        "Average P/L Ratio": average_p_l_ratio
    }