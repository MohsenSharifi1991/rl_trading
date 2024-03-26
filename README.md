# DRL for trading:

## 0) Baselines 
## 1) DQL
## 2) GP
## 3) GPBaseline
## 3) A2C



First step to use RL for any application is to define the problem. Then define the RL parametersts. Build framework, 
In Any RL problem we have the following:
problem statement: Can we develop and train a policy which can return maximum accumulated reward or return at the end of the day.
This mean that the agent should take actions (e.g. sell, hold, buy) stocks at different timelines such as (min, hour), then at the end of the day,
1) if return >= 100*x then, reward=+100
2) if 100x >return >10x then reward=+10
3) if return <= x then reward=-100

Agent: policy 
Reward: 
State: window of data as features St = (high t-3, high t-2, high t-1, ...)
Action: (-1,0,1)

State and evniroment are different. 

https://github.com/lukesalamone/deep-q-trading-agent
https://medium.com/@murrawang/deep-q-network-and-its-application-in-algorithmic-trading-16440a112e04