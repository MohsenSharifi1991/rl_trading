import numpy as np
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_cumulative_trade_return(daily_returns):
    """
    Plots the cumulative trade return based on daily returns DataFrame.

    Parameters:
    - daily_returns_df: A pandas DataFrame of daily returns with Date as the index.
    """
    # Ensure daily_returns is a DataFrame or Series
    if not isinstance(daily_returns, (pd.DataFrame, pd.Series)):
        raise TypeError("daily_returns must be a pandas DataFrame or Series.")

    # If daily_returns is a DataFrame, ensure the 'Daily Return' column exists
    if isinstance(daily_returns, pd.DataFrame) and 'Daily Reward' not in daily_returns.columns:
        raise ValueError("DataFrame must contain a 'Daily Reward' column.")

    # Calculate the cumulative returns
    if isinstance(daily_returns, pd.Series):
        cumulative_returns = (1 + daily_returns).cumprod() - 1
    else:  # it's a DataFrame
        cumulative_returns = (1 + daily_returns['Daily Reward']).cumprod() - 1

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(cumulative_returns.index, cumulative_returns, linewidth=2)

    plt.title('Cumulative Trade Return')
    plt.xlabel('Year')
    plt.ylabel('Cumulative Return')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Set x-axis major ticks to years
    plt.gca().xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))

    # Optional: rotate dates for better readability
    plt.gcf().autofmt_xdate()

    plt.tight_layout()
    plt.show()
