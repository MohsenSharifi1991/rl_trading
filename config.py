import torch
import datetime as dt
import os
curr_path = os.path.dirname(__file__)
# curr_time = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
curr_time = dt.datetime.now().strftime("%Y%m%d")

class Config:
    '''
    hyperparameters
    '''

    def __init__(self):
        ################################## env hyperparameters ###################################
        self.algo_name = 'DQN' # algorithmic name
        self.baseline_algo_names = ['LongOnly', 'SignR', 'MACDSignal']
        self.env_name = f'TradingSystem_v0_{self.algo_name}' # environment name
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")  # examine GPU
        self.seed = 11 # random seed
        self.train_eps = 200 # training episodes
        self.baseline_eps = 1
        ################################################################################

        ################################## algo hyperparameters ###################################
        # Common settings for all algorithms
        self.gamma = 0.3  # discount factor from paper
        self.lr = 0.0001  # learning rate for actor (PG and A2C) from paper

        # For DQN, set lr = 0.0001 (same as PG in the example) and add these
        self.lr_critic = 0.0001  # For A2C, learning rate for critic
        self.memory_capacity = 500 #5000  # For DQN, capacity of experience replay from paper
        self.target_update = 100 #1000  # For DQN, update frequency of target network from paper

        self.epsilon_start = 0.90  # start epsilon of e-greedy policy, adjust or remove based on algorithm
        self.epsilon_end = 0.01  # end epsilon of e-greedy policy, adjust or remove based on algorithm
        self.epsilon_decay = 500  # attenuation rate of epsilon in e-greedy policy, adjust or remove based on algorithm
        self.batch_size = 64  # size of mini-batch SGD from paper for PG and A2C

        # Hidden dimensions based on the LSTM network architecture from the paper
        # self.hidden_dim = [64, 32]  # dimensions of hidden layers, LSTM with 64 and 32 units
        self.hidden_dim = 8  # MLF

        # Transaction cost from paper, applicable in reward calculation
        self.transaction_cost_bp = 2  # basis points transaction cost from paper
        ################################################################################

        ################################# save path ##############################
        self.result_path = curr_path + "/outputs/" + self.env_name + \
                           '/' + curr_time + '/results/'
        self.model_path = curr_path + "/outputs/" + self.env_name + \
                          '/' + curr_time + '/models/'
        self.save = True  # whether to save the image