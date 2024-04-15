import pandas as pd
import numpy as np
from model import *
from market import *
from torch.utils.data import TensorDataset, DataLoader
from easydict import EasyDict
from utils import get_trajectories

"""
Preprocessing
"""
df = pd.read_csv('chart.csv', index_col=0)
df.index = pd.to_datetime(df.index)


date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='15min')
df = df.reindex(date_range, fill_value=np.nan)

df.close = df.close.ffill()
df.volume = df.volume.ffill()
df.value = df.value.ffill()

high = df.high.copy()
low = df.low.copy()
price = df.close.copy()

df.open /= df.close
df.high /= df.close
df.low /= df.close
df.close /= df.close.shift(1)
df.volume /= df.volume.shift(1)
df.value /= df.value.shift(1)

df = np.log(df)
df = df.fillna(0)

"""
Parameters
"""

config = EasyDict({
    'lr' : 1e-5,
    'batch_size' : 1024,
    'discount_factor' : 1.0,
    'gae_factor' : 0.9,
    'epoch' : 5,
    'clip' : 0.1,
    'n_market' : 128,
    'max_episode_steps' : 256,
    'obs_length' : 30,
    'buy_incentive' : 5e-3,
    'sell_incentive' : 5e-3,
    'incentive_decay' : 0.999,
    'epsilon' : 0.0,
    'trade_penalty' : -np.log(1-5e-4)

})

DEVICE = torch.device('cuda')


actor = Actor().to(DEVICE)
critic = Critic().to(DEVICE)

optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr = config.lr)
A2C = ActorCritic(actor, critic, config.clip)

env = MarketEnv(
    obs_length = config.obs_length,
    n_market = config.n_market,
    max_episode_steps=config.max_episode_steps,
    chart_df = df,
    buy_incentive= config.buy_incentive,
    sell_incentive= config.sell_incentive,
    incentive_decay= config.incentive_decay,
    epsilon = config.epsilon,
    trade_penalty = config.trade_penalty)

"""
Train
"""

while(True):

    trajectories = get_trajectories(env, A2C, config, DEVICE)
    sum_reward = trajectories[-1]
    print(sum_reward)
    trajectories = trajectories[:-1]
    dataset = TensorDataset(*trajectories)
    dataloader = DataLoader(dataset, config.batch_size, True)

    for i in range(config.epoch):
        for (idx, data) in enumerate(dataloader):
            surrogate_loss = A2C.surrogate_loss(*[d.to(DEVICE) for d in data])
            optimizer.zero_grad()
            surrogate_loss.backward()
            optimizer.step()