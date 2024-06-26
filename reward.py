def calculate_reward(action_t_minus_1, action_t_minus_2, delta_price_t,
                     sigma_t, sigma_t_minus_1, sigma_t_minus_2,
                     sigma_tgt, bp, price_t_minus_1):
    """
    Calculate the reward based on the trading action and market conditions.

    Parameters:
    - action_t_minus_1: The action taken at time t-1.
    - action_t_minus_2: The action taken at time t-2.
    - delta_price_t: The price change from t-1 to t.
    - sigma_t: The volatility at time t.
    - sigma_t_minus_1: The volatility at time t-1.
    - sigma_t_minus_2: The volatility at time t-2.
    - sigma_tgt: Target volatility for scaling.
    - bp: Transaction cost in basis points.
    - price_t_minus_1: The price at time t-1.

    Returns:
    - The calculated reward.
    """
    # Convert basis points to a proportion
    bp_cost = bp / 10000

    # Scale the actions by the ratio of volatilities
    scaled_action_t_minus_1 = action_t_minus_1 * (sigma_t_minus_1 / sigma_tgt)
    scaled_action_t_minus_2 = action_t_minus_2 * (sigma_t_minus_2 / sigma_tgt)

    # Calculate position-related reward, adjusted for volatility
    position_reward = scaled_action_t_minus_1 * delta_price_t / sigma_t

    # Calculate the transaction cost, adjusted for volatility scaling
    transaction_cost = abs(scaled_action_t_minus_1 - scaled_action_t_minus_2) * price_t_minus_1 * bp_cost

    # Total reward
    reward = position_reward - transaction_cost

    return reward

