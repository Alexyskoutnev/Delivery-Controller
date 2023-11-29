import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
import numpy as np
import argparse
from tensorboardX import SummaryWriter

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.algo.ppo import PPOAgent
from RL_Pizza_Delivery.utils.torch_utils import load_yaml, get_device, save_model, save_frames

def train(agent, env, config, writer, device='cpu'):
    global_step = 0
    #Storage Setup
    obs = torch.zeros((config['num_steps'],) + (env.observation_dim,), dtype=torch.float32).to(device)
    actions = torch.zeros((config['num_steps'],) + (1,), dtype=torch.float32).to(device)
    logprobs = torch.zeros((config['num_steps'],), dtype=torch.float32).to(device)
    rewards = torch.zeros((config['num_steps'],), dtype=torch.float32).to(device)
    dones = torch.zeros((config['num_steps'],), dtype=torch.float32).to(device)
    values = torch.zeros((config['num_steps'],), dtype=torch.float32).to(device)
    next_obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
    next_done = torch.zeros(1, dtype=torch.float32).to(device)
    average_returns = []
    #Storage Setup
    num_updates = config['total_timestep'] // config['batchs']
    for update in range(1, num_updates + 1):
        total_reward = 0.0
        for step in range(0, config['num_steps']):
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
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                returns = torch.zeros_like(rewards).to(device)
                for t in reversed(range(config['num_steps'])):
                    if t == config['num_steps'] - 1:
                        nextnonterminal = 1.0 - next_done
                        next_return = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        next_return = returns[t + 1]
                    returns[t] = rewards[t] + config['gamma'] * nextnonterminal * next_return
                advantages = returns - values
        
        average_returns.append(total_reward)
        #================ Batch of Experience ===================
        b_obs = obs.reshape((-1, ) + (env.observation_dim,))
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + (1,))
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)
        b_indx = np.arange(config['batchs'])
        #================ Batch of Experience ===================
        v_loss, pg_loss, loss = agent.update(b_obs, b_logprobs, b_actions, b_advantages, b_returns, b_values)
        #================ Logging ================ 
        print(f"[{global_step}] Mean Reward: {np.mean(average_returns[-100:]):.3f}")
        print(f"[{global_step}] Value Loss : {v_loss.item():.3f}")
        print(f"[{global_step}] Critic Loss : {pg_loss.item():.3f}")
        #================ Logging ================ 
        if update % config['EVAL_ITR'] == 0:
            state = env.reset()
            buf = list()
            eval_total_rewards = 0
            for i in range(100):
                if config['render_mode'] == 'rgb_array':
                    buf.append(env.render())
                action, prob, entropy, value = agent.get_action_and_value(torch.tensor(state, dtype=torch.float32).view(1, -1))
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                state = next_state
                eval_total_rewards += reward
                if done:
                    buf.append(env.render())
                    break
            print("====================== EVALUALTION ======================")
            print(f"{global_step}: Epoch [{update}/{config['epochs']}] : reward [{eval_total_rewards:.3f}] \t")
            if config['record_vid']:
                save_frames(buf, name="PPO_")
    save_model(agent.critic, config, name="PPO-critic")
    save_model(agent.actor, config, name="PPO-actor" )
    env.close()
    writer.close()

if __name__ == "__main__":
    writer = SummaryWriter(comment='-PPO')
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="PPO_MAP_5_5_HOLES_0.yaml", help="Path to a training config in /data/config")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config['device'] = get_device()
    device = get_device()
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    agent = PPOAgent(env, config)
    train(agent, env, config, writer, device)