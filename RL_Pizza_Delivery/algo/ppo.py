import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, env):
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
    
    def update():
        pass

if __name__ == "__main__":
     # Training parameters
    potholes, traffic_jams = 0, 0
    map_size = (5, 5)
    render_mode = 'None'
    lr = 1e-3
    epochs = 1000
    num_steps = 100
    total_timestep = int(1e5)
    # total_timestep = epochs * num_step
    batchs = 32
    device = 'cpu'
    gamma = 0.99

    env = ENV_OBSTACLE(map_size=map_size, render_mode=render_mode, potholes=potholes)
    agent = PPOAgent(env)
    optimizer = optim.Adam(agent.parameters(), lr, eps=1e-5)

    global_step = 0

    #Storage Setup
    obs = torch.zeros((total_timestep) + env.observation_dim, dtype=torch.float32).to(device)
    actions = torch.zeros((total_timestep) + env.action_dim, dtype=torch.float32).to(device)
    logprobs = torch.zeros((total_timestep), dtype=torch.float32).to(device)
    rewards = torch.zeros((total_timestep), dtype=torch.float32).to(device)
    dones = torch.zeros((total_timestep), dtype=torch.float32).to(device)
    values = torch.zeros((total_timestep), dtype=torch.float32).to(device)
    

    next_obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    next_done = torch.zeros(1, dtype=torch.float32).to(device)
    num_updates = total_timestep // batchs


    for update in range(1, num_updates + 1):
        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_done
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, done,  _ = env.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs = torch.tensor(next_obs, dtype=torch.float32).to(device)
            next_done = torch.tensor(done, dtype=torch.int).to(device)

            #Bootstrap the return 
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(num_steps)):
                    if t == num_steps - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + gamma * nextnonterminal * next_return
                advantages = returns - values
        pass