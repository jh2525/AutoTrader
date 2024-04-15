
import torch
import numpy as np
import pandas as pd
from enum import Enum

class State(Enum):
    WaitAndSee = 0
    Hold = 1
    Done = 2


class Market:
    
    def __init__(self, chart_data: pd.DataFrame, obs_length = 30, index = 0, trade_penalty = -np.log(1-5e-4), display = False, buy_incentive = 0.01, sell_incentive = 0.01, epsilon = 0.0):
        self.char_data: pd.DataFrame = chart_data
        self.index: int = index + obs_length - 1
        self.trade_penalty: float = trade_penalty
        self.obs_length: int = obs_length
        self.buy_incentive: float = buy_incentive
        self.sell_incentive: float = sell_incentive
        self.truncated: bool = False
        self.state: State = State.WaitAndSee
        self.display: bool = display
        self.epsilon: float = epsilon
    
    def get_observation(self):
        return self.char_data.iloc[self.index - self.obs_length + 1:self.index + 1].to_numpy()

    def buy(self, theta: float):

        if self.display:
            print(f'[{self.char_data.index[self.index]}] : buy {theta}')

        self.index += 1

        c_t1, l_t1 = self.char_data.iloc[self.index].loc[['close', 'low']]

        if theta - self.epsilon > l_t1 + c_t1:
            if self.display:
                print(f'[{self.char_data.index[self.index]}] : executed a buy order at {theta}. high : {l_t1}, close : {c_t1}')
            self.state = State.Hold
            return c_t1 - theta - self.trade_penalty + self.buy_incentive
        else:
            if self.display:
                print(f'[{self.char_data.index[self.index]}] : buy order is failed')
            return 0.0

    def sell(self, theta: float):

        if self.display:
            print(f'[{self.char_data.index[self.index]}] : sell {theta}')

        self.index += 1

        c_t1, h_t1 = self.char_data.iloc[self.index].loc[['close', 'high']]

        if theta + self.epsilon < c_t1 + h_t1:
            if self.display:
                print(f'[{self.char_data.index[self.index]}] : executed a sell order at {theta}. low : {h_t1}, close : {c_t1}')
            
            self.state = State.Done
            return -c_t1 + theta - self.trade_penalty + self.sell_incentive
        else:
            if self.display:
                print(f'[{self.char_data.index[self.index]}] : sell order is failed')
            return c_t1
            
    def hold(self):

        self.index += 1

        c_t1 = self.char_data.iloc[self.index].loc['close']

        if self.state is State.Hold:
            return c_t1
        else:
            return 0.0

    def get_available_action(self):
        if self.state is State.WaitAndSee:
            return [True, False, True]
        elif self.state is State.Hold:
            return [False, True, True]
        else:
            return [False, False, False]

    def step(self, action, theta):
        self.truncated = self.index >= len(self.char_data) - 1
        
        if self.state is State.Done or self.truncated:
            return np.zeros((self.obs_length, 6)), 0.0, self.state is State.Done, self.truncated, {}

        if action == 0:
            reward = self.buy(theta)
        elif action == 1:
            reward = self.sell(theta)
        else:
            reward = self.hold()
        
        next_observation = self.get_observation()
        return next_observation, reward, (self.state is State.Done), self.truncated, {}

class MarketEnv():
    def __init__(self, obs_length, n_market, max_episode_steps, chart_df, buy_incentive, sell_incentive, incentive_decay, epsilon, trade_penalty):
        self.n_market = n_market
        self.obs_length = obs_length
        self.max_episode_steps = max_episode_steps
        self.df = chart_df
        self.buy_incentive = buy_incentive
        self.sell_incentive = sell_incentive
        self.incentive_decay = incentive_decay
        self.reset_count = 0
        self.epsilon = epsilon
        self.trade_penalty = trade_penalty

    def reset(self):
        start_idx = np.random.randint(0, len(self.df) - self.max_episode_steps - self.obs_length, size = self.n_market)
        self.markets = [Market(
            obs_length = self.obs_length,
            chart_data = self.df.iloc[i:i+self.max_episode_steps + self.obs_length], 
            display = False, 
            epsilon = self.epsilon,
            buy_incentive = self.buy_incentive * (self.incentive_decay ** self.reset_count),
            sell_incentive = self.sell_incentive * (self.incentive_decay ** self.reset_count),
            trade_penalty = self.trade_penalty) 
            for i in start_idx]
        
        self.reset_count += 1
        return self.get_obs(), self.get_mask()

    def get_obs(self):
        return torch.from_numpy(np.stack([self.markets[i].get_observation() for i in range(self.n_market)])).float()
    
    def get_mask(self):
        return torch.from_numpy(np.stack([self.markets[i].get_available_action() for i in range(self.n_market)]))
    
    def step(self, actions, action_values):
        next_obses = []
        rewards = []
        dones = []
        truncated = []
        infos = []
        masks = []
        

        for i in range(self.n_market):
            if not (self.markets[i].state is State.Done or self.markets[i].truncated):
                infos.append(i)
            else:
                continue
            

            next_obs, reward, done, trunacted, info = self.markets[infos[-1]].step(actions[len(infos) - 1].item(), action_values[len(infos) - 1].item())
            next_obses.append(next_obs)
            rewards.append(reward)
            dones.append(done)
            truncated.append(trunacted)
            masks.append(self.markets[i].get_available_action())
            
        if len(infos) == 0:
            return None, None, None, None, None, None

        next_obses = torch.from_numpy(np.stack([next_obses[i] for i in range(len(infos))])).float()
        rewards = torch.from_numpy(np.array([rewards[i] for i in range(len(infos))])).float()
        dones = torch.from_numpy(np.array([dones[i] for i in range(len(infos))]))
        truncated = dones = torch.from_numpy(np.array([truncated[i] for i in range(len(infos))]))
        masks = torch.from_numpy(np.stack([masks[i] for i in range(len(infos))])).bool()

        return next_obses, rewards, dones, truncated, masks, infos
        
