import torch.nn as nn
import torch
from torch.masked import masked_tensor, as_masked_tensor

class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()

        self.hidden_size = 256
        self.num_feature = 6
        self.num_layer = 1
        self.gru = nn.GRU(self.num_feature, hidden_size = self.hidden_size, num_layers=self.num_layer, batch_first=True)


        self.std_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Softplus()
        )
        self.myu_layer = nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
        self.logit_layer =  nn.Sequential(
            nn.Linear(self.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

        self.myu_layer[-1].weight.data /= 100
        self.myu_layer[-1].bias.data /= 100
        self.std_layer[-2].bias.data -= 5
    
    def forward(self, x):
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(x.device)
        _, h_n = self.gru(x, h0)
        outlet = h_n[-1]

        stds = self.std_layer(outlet)
        myus = self.myu_layer(outlet)
        logits = self.logit_layer(outlet)

        return myus, stds, logits
    

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()

        self.hidden_size = 256
        self.num_feature = 6
        self.num_layer = 1
        self.gru = nn.GRU(self.num_feature, hidden_size = self.hidden_size, num_layers=self.num_layer, batch_first=True)


        self.was_value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

        self.hold_value_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )
        
    
    def forward(self, x, action_mask):
        h0 = torch.zeros(self.num_layer, x.size(0), self.hidden_size).to(x.device)
        _, h_n = self.gru(x, h0)
        outlet = h_n[-1]
        was_values = self.was_value_layer(outlet)
        hold_value = self.hold_value_layer(outlet)

        return was_values * action_mask[:, [0]] + hold_value * (~action_mask[:, [0]])
    

class ActorCritic():
    def __init__(self, actor, critic, clip):
        self.actor = actor
        self.critic = critic
        self.clip = clip

    
    def sample(self, obs, action_mask):
        with torch.no_grad():
            myus, stds, logits = self.actor(obs)
            state_values = self.critic(obs, action_mask)

            myus, stds, logits, state_values = [x.detach() for x in [myus, stds, logits, state_values]]
        logits = torch.where(action_mask, logits, -1e36)

        pdf = torch.distributions.Normal(myus, stds)
        action_categorical = torch.distributions.Categorical(logits = logits)

        value_sample = pdf.sample()
        action_sample = action_categorical.sample()

        hold_value_log_probs = torch.zeros_like(state_values)


        
        value_log_probs = torch.concat([pdf.log_prob(value_sample), hold_value_log_probs], dim=-1)
        log_probs = value_log_probs.gather(dim=-1, index = action_sample.unsqueeze(dim=1)).flatten() + action_categorical.log_prob(action_sample).flatten()
        value_sample = torch.concat([value_sample, hold_value_log_probs], dim=-1).gather(dim=-1, index = action_sample.unsqueeze(dim=1)).flatten()

        return value_sample.cpu(), action_sample.cpu(), log_probs.cpu(), state_values.cpu()


    def surrogate_loss(self, target_state_value, advantages, old_log_probs, obs, actions, value_actions, masking):


        myus, stds, logits = self.actor(obs)
        state_values = self.critic(obs, masking)

        logits = torch.where(masking, logits, -1e36)

        pdf = torch.distributions.Normal(myus, stds)
        action_categorical = torch.distributions.Categorical(logits = logits)

        hold_value_log_probs = torch.zeros_like(state_values)

        value_actions = value_actions.unsqueeze(dim=-1).expand((value_actions.size(0), 2))


        value_log_probs = torch.concat([pdf.log_prob(value_actions), hold_value_log_probs], dim=-1)
        log_probs = value_log_probs.gather(dim=-1, index = actions.unsqueeze(dim=-1)).flatten() + action_categorical.log_prob(actions).flatten()


        prob_ratio = torch.exp(log_probs - old_log_probs)
        critic_loss = (state_values - target_state_value).square().mean()

        actor_loss = -torch.min(
            advantages * prob_ratio, advantages * torch.clip(prob_ratio, 1-self.clip, 1+self.clip)
        ).mean()

        return critic_loss + actor_loss

