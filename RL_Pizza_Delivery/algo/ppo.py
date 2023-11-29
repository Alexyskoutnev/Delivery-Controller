import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from tensorboardX import SummaryWriter

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, env, config=None):
        super().__init__()
        self.obs_size = env.observation_dim
        self.action_size = env.action_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
            layer_init(nn.Linear(64, self.action_size), std=0.01)
        )
        self.update_epoch = config['update_epochs']
        self.batchs = config['batchs']
        self.mini_batch = config['mini_batch']
        self.clipping_coef = config['clipping_coef']
        self.entropy_coef = config['entropy_coef']
        self.optimizer = optim.Adam(self.parameters(), config['lr'], eps=1e-5)
    
    def get_value(self, x):
        return self.critic(x)

    def compute_returns(self, rewards, next_value):
        returns = []
        R = next_value
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def get_action_and_value(self, x, action=None):
        _probs = F.softmax(self.actor(x), dim=-1)
        probs = Categorical(probs=_probs)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def update(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        b_indx = np.arange(self.batchs)
        for epoch in range(self.update_epoch):
            np.random.shuffle(b_indx)
            for start in range(0, self.batchs, self.mini_batch):
                end = start + self.mini_batch
                idxs = b_indx[start:end]
                _, new_logprob, entropy, new_value = self.get_action_and_value(b_obs[idxs], b_actions[idxs])
                logratio = new_logprob - b_logprobs[idxs]
                ratio = logratio.exp()
                mb_advantages = b_advantages[idxs]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                #============= Policy Loss =================
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clipping_coef, 1 + self.clipping_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                #============= Policy Loss =================
                #============= Critic Loss =================
                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - b_returns[idxs])** 2).mean()
                entropy_loss = entropy.mean()
                #============= Critic Loss =================
                #============= Update Step =================
                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                #============= Update Step =================
        return v_loss, pg_loss, loss

if __name__ == "__main__":
     # Training parameters
    writer = SummaryWriter(comment='-PPO')
    #================== Config ================== 
    potholes, traffic_jams = 0, 0
    map_size = (3, 3)
    render_mode = 'None'
    lr = 1e-3
    epochs = 1000
    num_steps = 64
    total_timestep = int(1e7)
    batchs = 8
    mini_batch = 4
    device = 'cpu'
    gamma = 0.99
    update_epochs = 4 #Standard Values
    clipping_coef = 0.1 #Standard Values
    entropy_coef = 0.1 #Standard Values
    max_grad_norm = 0.5 #Standard Values
    config = {"update_epochs": update_epochs, "batchs": batchs, "mini_batch": mini_batch,
            "clipping_coef" : clipping_coef, "entropy_coef" : entropy_coef}
    #================== Config ================== 
    env = ENV_OBSTACLE(map_size=map_size, render_mode=render_mode, potholes=potholes)
    agent = PPOAgent(env, config)
    optimizer = optim.Adam(agent.parameters(), lr, eps=1e-5)