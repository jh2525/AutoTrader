import torch

def get_trajectories(env, A2C, config, DEVICE):
    next_obs, next_mask = env.reset()
    trajectories = dict(zip(range(env.n_market), [{'action_values':[], 'actions':[], 'log_probs':[], 'state_values':[], 'masks':[], 'rewards':[], 'obses':[]} for i in range(env.n_market)]))
        
    while(True):
        obs = next_obs.to(DEVICE)
        mask = next_mask.to(DEVICE)
        action_values, actions, log_probs, state_values =  A2C.sample(obs, mask)

        
        next_obs, reward, done, truncated, next_mask, info = env.step(actions, action_values)
        if next_obs is None:
            break

        for j in range(len(info)):
            trajectories[info[j]]['action_values'].append(action_values[j])
            trajectories[info[j]]['actions'].append(actions[j])
            trajectories[info[j]]['log_probs'].append(log_probs[j])
            trajectories[info[j]]['state_values'].append(state_values[j])
            trajectories[info[j]]['obses'].append(obs[j])
            trajectories[info[j]]['rewards'].append(reward[j])
            trajectories[info[j]]['masks'].append(mask[j])

    total_reward = 0.0

    for i in trajectories:
        trajectories[i]['action_values'] = torch.Tensor(trajectories[i]['action_values'])
        trajectories[i]['actions'] = torch.Tensor(trajectories[i]['actions'])
        trajectories[i]['log_probs'] = torch.Tensor(trajectories[i]['log_probs'])
        trajectories[i]['state_values'] = torch.Tensor(trajectories[i]['state_values'])
        trajectories[i]['obses'] = torch.stack(trajectories[i]['obses'])
        trajectories[i]['rewards'] = torch.Tensor(trajectories[i]['rewards'])
        trajectories[i]['masks'] = torch.stack(trajectories[i]['masks'])


        estimated_V = trajectories[i]['state_values']
        rewards = trajectories[i]['rewards']
        

        GAE = torch.zeros((len(rewards),))

        delta_t = - estimated_V[-1] + rewards[-1]
        GAE[-1] = delta_t

        for j in reversed(range(len(GAE) - 1)):
            delta_t = -estimated_V[j] + rewards[j] + config.discount_factor*estimated_V[j+1]
            GAE[j] = config.discount_factor * config.gae_factor * GAE[j+1] + delta_t

        trajectories[i]['target_state_value'] = GAE + estimated_V
        trajectories[i]['advantages'] = GAE

        total_reward += trajectories[i]['rewards'].sum().item()
        
    #target_state_value, advantages, old_log_probs, obs, actions, value_actions, masking

    target_state_value = torch.concat([
        trajectories[i]['advantages'] for i in trajectories
    ])
    advantages = torch.concat([
        trajectories[i]['advantages'] for i in trajectories
    ])
    old_log_probs = torch.concat([
        trajectories[i]['log_probs'] for i in trajectories
    ])
    obs = torch.concat([
        trajectories[i]['obses'] for i in trajectories
    ])
    actions = torch.concat([
        trajectories[i]['actions'] for i in trajectories
    ]).long()
    value_actions = torch.concat([
        trajectories[i]['action_values'] for i in trajectories
    ])
    masking = torch.concat([
        trajectories[i]['masks'] for i in trajectories
    ]).bool()

    return target_state_value, advantages, old_log_probs, obs, actions, value_actions, masking, total_reward
