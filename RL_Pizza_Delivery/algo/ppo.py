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
                end = start + mini_batch
                idxs = b_indx[start:end]
                _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[idxs], b_actions[idxs])
                logratio = new_logprob - b_logprobs[idxs]
                ratio = logratio.exp()
                mb_advantages = b_advantages[idxs]
                mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
                #============= Policy Loss =================
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.clipping_coef, 1 +clipping_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()
                #============= Policy Loss =================
                #============= Critic Loss =================
                new_value = new_value.view(-1)
                v_loss = 0.5 * ((new_value - b_returns[idxs])** 2).mean()
                entropy_loss = entropy.mean()
                #============= Critic Loss =================
                #============= Update Step =================
                loss = pg_loss - self.entropy_coef * entropy_loss + v_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
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
    num_steps = 32
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
    global_step = 0
    #Storage Setup
    obs = torch.zeros((num_steps,) + (env.observation_dim,), dtype=torch.float32).to(device)
    actions = torch.zeros((num_steps,) + (1,), dtype=torch.float32).to(device)
    logprobs = torch.zeros((num_steps,), dtype=torch.float32).to(device)
    rewards = torch.zeros((num_steps,), dtype=torch.float32).to(device)
    dones = torch.zeros((num_steps,), dtype=torch.float32).to(device)
    values = torch.zeros((num_steps,), dtype=torch.float32).to(device)
    next_obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    next_done = torch.zeros(1, dtype=torch.float32).to(device)
    average_returns = []
    
    #Storage Setup
    num_updates = total_timestep // batchs
    for update in range(1, num_updates + 1):
        total_reward = 0.0
        for step in range(0, num_steps):
            global_step += 1
            obs[step] = next_obs
            dones[step] = next_done
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob
            next_obs, reward, done,  _ = env.step(action.cpu().numpy())
            total_reward += reward
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
        
        average_returns.append(total_reward)
        #================ Batch of Experience ===================
        b_obs = obs.reshape((-1, ) + (env.observation_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_indx = np.arange(batchs)
        #================ Batch of Experience ===================
        clip = []
        v_loss, pg_loss, loss = agent.update(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)

        # for epoch in range(update_epochs):
        #     np.random.shuffle(b_indx)
        #     for start in range(0, batchs, mini_batch):
        #         end = start + mini_batch
        #         idxs = b_indx[start:end]
        #         _, new_logprob, entropy, new_value = agent.get_action_and_value(b_obs[idxs], b_actions[idxs])
        #         logratio = new_logprob - b_logprobs[idxs]
        #         ratio = logratio.exp()
        #         with torch.no_grad():
        #             old_approx_kl = (-logprob).mean()
        #             approx_kl = ((ratio -1) - logprob).mean()
        #             clip += [((ratio - 1.0).abs() > clipping_coef).float().mean().item()]
                
        #         mb_advantages = b_advantages[idxs]
        #         mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)
        #         #============= Policy Loss =================
        #         pg_loss1 = -mb_advantages * ratio
        #         pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clipping_coef, 1 +clipping_coef)
        #         pg_loss = torch.max(pg_loss1, pg_loss2).mean()
        #         #============= Policy Loss =================
        #         #============= Critic Loss =================
        #         new_value = new_value.view(-1)
        #         v_loss = 0.5 * ((new_value - b_returns[idxs])** 2).mean()
        #         entropy_loss = entropy.mean()
        #         #============= Critic Loss =================
        #         #============= Update Step =================
        #         loss = pg_loss - entropy_coef * entropy_loss + v_loss
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         #============= Update Step =================
        print(f"[{global_step}] Mean Reward: {np.mean(average_returns[-100:]):.3f}")
        print(f"[{global_step}] Value Loss : {v_loss.item():.3f}")
        print(f"[{global_step}] Critic Loss : {pg_loss.item():.3f}")
        #=================== Logging =======================================
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        # writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        # writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        # writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        # writer.add_scalar("losses/clipfrac", np.mean(clip), global_step)    

    env.close()
    writer.close()    

                
                


                

        

