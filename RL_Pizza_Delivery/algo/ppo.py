import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
from tensorboardX import SummaryWriter

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    Initialize the weights and bias of a neural network layer.

    Parameters:
    - layer: The neural network layer to be initialized.
    - std: The standard deviation for weight initialization. Defaults to np.sqrt(2).
    - bias_const: The constant value for bias initialization. Defaults to 0.0.

    Returns:
    - layer: The initialized layer.
    """
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class PPOAgent(nn.Module):
    def __init__(self, env, config=None):
        """
        Proximal Policy Optimization (PPO) agent class.

        Parameters:
        - env: The environment in which the agent operates.
        - config (dict, optional): Configuration parameters for the agent. Defaults to None.
        """
        super().__init__()
        self.obs_size = env.observation_dim
        self.action_size = env.action_dim
        self.critic = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(self.obs_size, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, self.action_size), std=0.01)
        )
        self.update_epoch = config['update_epochs']
        self.batchs = config['batchs']
        self.mini_batch = config['mini_batch']
        self.clipping_coef = config['clipping_coef']
        self.entropy_coef = config['entropy_coef']
        self.optimizer = optim.Adam(self.parameters(), config['lr'], eps=1e-5)
    
    def get_value(self, x : torch.Tensor) -> torch.Tensor:
        """Get the critic value based on the sate

        Args:
            x (torch.Tensor): Current state that the agent is in

        Returns:
            torch.Tensor: The value based on the state input
        """
        return self.critic(x)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        """Forward function for the neural network

        Args:
            x (torch.Tensor): Current state that the agent is in

        Returns:
            torch.Tensor: Action from the agent
        """
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        action = probs.sample()
        return action
        
    def compute_returns(self, rewards : torch.Tensor, next_value : torch.Tensor):
        """
        Compute the returns for each time step given a list of rewards.

        Parameters:
        - rewards (list): List of rewards obtained at each time step.
        - next_value: The value of the next state.

        Returns:
        - torch.FloatTensor: Returns for each time step.
        """
        returns = []
        R = next_value
        for reward in reversed(rewards):
            R = reward + self.gamma * R
            returns.insert(0, R)
        return torch.FloatTensor(returns)

    def get_action_and_value(self, x, action=None):
        """
        Get the action, log probability, entropy, and value for a given state.

        Parameters:
        - x: The input state.
        - action: The chosen action. If None, a new action is sampled.

        Returns:
        - tuple: Tuple containing the action, log probability, entropy, and value.
        """
        _probs = F.softmax(self.actor(x), dim=-1)
        logits = self.actor(x)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        else:
            action = torch.squeeze(action, dim=1)
        return action, probs.log_prob(action), probs.entropy(), self.critic(x)
    
    def update(self, b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values):
        """
        Update the PPO agent using the Proximal Policy Optimization algorithm.

        Parameters:
        - b_obs: Batch of observations.
        - b_logprobs: Batch of log probabilities.
        - b_actions: Batch of actions.
        - b_advantages: Batch of advantages.
        - b_returns: Batch of returns.
        - b_values: Batch of values.

        Returns:
        - tuple: Tuple containing the value loss, policy gradient loss, and total loss.
        """
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