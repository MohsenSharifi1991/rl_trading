class LongOnly:
    def __init__(self, action_dim):
        self.action_dim = action_dim  # Assuming action_dim includes a 'long' action

    def choose_action(self, state):
        action = 2  # Always long
        return action, []


class SignR:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def choose_action(self, state):
        # Index for 'return_252d' based on the provided state columns is 11
        past_return = state[11]  # Use 'return_252d' for decision
        action = 1 if past_return > 0 else -1  # Long if 'return_252d' is positive, otherwise short
        return action, []


class MACDSignal:
    def __init__(self, action_dim):
        self.action_dim = action_dim

    def choose_action(self, state):
        macd_signal = state[12]  # Assuming 'macd' is at index 12 based on your state columns
        action = 1 if macd_signal > 0 else -1  # Long if MACD signal is positive, short if negative
        return action, []


