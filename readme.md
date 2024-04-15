# AutoTrader

This is AutoTrader (for Cryptocurrency, Stock, ...) using the PPO algorithm in Reinforcement Learning.

The model and parameters of this project are just an example. If you want to improve performance, consider structuring your own model.

If you are interested in the algorithm used in this project, please refer to the following papers:  
[Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)  
[High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438)

# Problems of Algorithm Trading using Deep Learning

There are several ways to use deep learning for trading, but the most basic method is to predict the price after a certain period and purchase the stock if a specific increase rate is expected. However, even if the price is predicted using this method and assuming the predicted price is accurate, the following problems exist:

**1. It is not always guaranteed to purchase the stock at the same price as the closing price.**  
Let's assume that the closing price of a stock A is 1000 at time $t_1$. And let's assume that the closing price of A at time $t_2$ is predicted to be 1100 won. If you can purchase the stock, you can expect a 10% return. However, it is unreasonable to assume that you can always buy the stock at that price. For example, if the ask price is at 1050 and the bid price is at 1000, you cannot always purchase the stock at that price. Therefore, even if there is a clear buy signal, it is very difficult to figure out how to purchase the stock and whether it will generate a profit if purchased that way.

**2. Fee issue**  
In actual stock trading, fees are incurred. Therefore, to actually earn a profit through trading, the expected return must be calculated considering the fees.

The above two problems can be easily solved by using a reinforcement learning algorithm. By predicting the ask and bid prices between time $t_1$ and $t_2$ based on the price at time $t_1$ (continuous action) and placing orders, it can be resolved. If the high price is greater than the ask price, it can be assumed that the sell order was executed, and if the bid price is greater than the low price, it can be assumed that the buy order was executed. These assumptions are reasonable unless a very large amount is traded. Using reinforcement learning, this process can be learned very efficiently. Furthermore, since reinforcement learning aims to maximize the expected sum of rewards, there is no need to use a specific algorithm for trading. (If the log return is given as a reward, the sum of those rewards exactly matches the log return of the final return.)

# Environment

## Action

There are three actions as follows ($c_t$ is the closing price at the current time $t$, $h_t$ is the high price, $l_t$ is the low price, and $\theta$ is a parameter):

| Num | Action      | Control Range | Description                                              |
|-----|-------------|---------------|----------------------------------------------------------|
| 0   | Buy         | (-inf, inf)   | Place a buy order at the price of $e^{\theta}c_t$.       |
| 1   | Sell        | (-inf, inf)   | Place a sell order at the price of $e^{\theta}c_t$.      |
| 2   | Do nothing  |               |                                                          |

If the chosen action is Buy or Sell, you need to provide trading parameters. Also, if $e^{\theta}c_t < h_{t+1}$, it is considered a successful buy, and if $e^{\theta}c_t > l_{t+1}$, it is considered a successful sell. This is a reasonable way to check the success of trading assuming the trading volume is sufficiently small.

## State

| State        | Description                                                                   |
|--------------|-------------------------------------------------------------------------------|
| Wait-and-see | The state until the buy order is executed                                    |
| Hold         | The state after the buy order is executed and until the sell order is executed |
| Done         | The state after both buy and sell orders are executed. The episode ends when this state is reached. |

## Reward

Rewards are given according to the state and action as follows. Let's denote the closing price at time $t$ as $c_t$ and the trading penalty (e.g., fees) as $p$.

1. **Wait-and-see**  
**(1) When Buy is taken and executed**  
$r_t = \log\frac{c_{t+1}}{\text{buy price}}$ - $p$  
**(2) When Buy is taken but not executed**  
Since it is the same as not taking any action, a reward of 0 is obtained.  
**(3) Nothing**  
Since there is no change in assets, a reward of 0 is obtained.  
1. **Hold**  
**(1) When Sell is taken and executed**  
$r_t = \log\frac{\text{sell price}}{c_{t+1}}$ - $p$   
**(2) When Sell is taken but not executed**  
Since it is the same as not taking any action, the change in the held assets is received as a reward.  
 $r_t = \log\frac{c_{t+1}}{c_t}$  
**(3) Do Nothing**  
 $r_t = \log\frac{c_{t+1}}{c_t}$  

If the rewards are given as above, it can be easily shown that the sum of rewards in an episode where buy and sell occur is $\log\frac{\text{sell price}}{\text{buy price}} -2p$. **The sum of rewards exactly matches the actual log return.**

Let's briefly show this fact. Assume that a buy occurs at time $t_b$ and a sell occurs at time $t_s$. Then,

$$\log\frac{\text{sell price}}{\text{buy price}} = \log\left[\frac{c_{t_b+1}}{\text{buy price}} \cdot \frac{c_{t_b+2}}{c_{t_b+1}} \cdot \frac{c_{t_b+3}}{c_{t_b+2}} \cdots \frac{c_{t_s+1}}{c_{t_s}}\cdot \frac{\text{sell price}}{{c_{t_s+1}}}\right] = \log{\frac{c_{t_b+1}}{\text{buy price}}} + \log\frac{c_{t_b+2}}{c_{t_b+1}} + \cdots + \frac{\text{sell price}}{{c_{t_s+1}}}$$

Therefore, it is self-evident.

## Preprocessing
When chart data is given, an observation is created through the preprocessed chart dataframe as follows:

- $h^\*_t = \log h_t - \log c_t$
- $l^\*_t = \log l_t - \log c_t$
- $c^\*_t = \log c_t - \log c\_\{t-1\}$

In other words, in this project, as long as there is a dataframe series that satisfies the above, any feature can be used and well-compatible.  
Preprocessing in this way has the following advantages:

- **It becomes easier to calculate rewards and check the execution of trading.**  
For example, let's assume that a buy is successfully executed with parameter $\theta$ at time $t$. Then the reward is given as $\log( c_{t+1} / c_te^\theta)$, which becomes $\log( c_{t+1} / c_te^\theta) = \log c_{t+1} - \log c_t - \theta = c_{t+1}^* - \theta$, and the reward for the nothing action in the hold state simply becomes $c_{t+1}^\*$. Also, the execution of trading can be easily checked. For example, in the case of a buy, the condition is $e^{\theta}c_t > l_{t+1}$. Taking the logarithm, it becomes $\theta + \log c_t > \log l_{t+1}$, which can be easily checked as $\theta + \log c_t - \log c_{t+1} = \theta - c_{t+1}^\* > \log l_{t+1} - \log c_{t+1} = l^\*_{t+1}$.
- **It makes the data stationary.**  
Since calculations are made only based on ratios regardless of the absolute size of prices, it makes the chart data stationary.

## Parameters

| Parameter         | Type  | Default | Description  |
|-------------------|-------|---------|--------------|
| lr                | float | 1e-5    | Learning rate |
| batch_size        | int   | 1024    |              |
| discount_factor   | float | 1.0     |              |
| gae_factor        | float | 0.9     |              |
| epoch             | int   | 5       |              |
| clip              | float | 1e-1    |              |
| n_market          | int   | 128     | The number of trajectories |
| max_episode_steps | int   | 256     |              |
| obs_length        | int   | 30      |              |
| buy_incentive     | float | 5e-3    | Additional reward given when a buy is executed |
| sell_incentive    | float | 5e-3    | Additional reward given when a sell is executed |
| incentive_decay   | float | 0.999   | Decay rate of trading incentives |
| epsilon           | float | 0.0     | Parameter that controls the execution of trading (the higher it is, the lower or higher the price needs to be compared to the high or low price for execution) |
| trade_penalty     | float | -log(1-5e-4) | Penalty received when trading is executed |

# Training
To eliminate time dependence, trajectories are obtained by selecting random time points and trading only for a certain period of time.

![image](https://github.com/jh2525/AutoTrader/assets/160830734/816266c4-9b8a-4745-959c-92613a6c0aae)

Also, unlike a typical policy, it is a method of first choosing a discrete action and then taking a continuous action according to a specific action. Considering that the policy can be calculated as follows, the probability can be easily obtained.

$\pi(\theta, \text{Buy} | s) = \pi(\theta | \text{Buy}, s) \pi(\text{Buy} | s)$  
$\pi(\theta, \text{Sell} | s) = \pi(\theta | \text{Sell}, s) \pi(\text{Sell} | s)$

Please refer to the code for more details.