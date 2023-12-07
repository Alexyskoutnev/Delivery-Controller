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

def train(agent : PPOAgent, env : ENV_OBSTACLE, config : dict,
          writer : SummaryWriter, device : dict = 'cpu'):
    """
    Train a Proximal Policy Optimization (PPO) agent in the specified environment.

    Parameters:
    - agent (PPOAgent): The PPO agent to be trained.
    - env (ENV_OBSTACLE): The environment in which the agent is trained.
    - config (dict): Configuration parameters for training.
    - writer (SummaryWriter): TensorBoard SummaryWriter for logging.
    - device (str, optional): The device on which the training is performed ('cpu' or 'cuda'). Defaults to 'cpu'.

    Returns:
    None
    """
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
    average_rewards = []
    average_return = []
    #Storage Setup
    num_updates = config['total_timestep'] // config['batchs']
    for update in range(1, num_updates + 1):
        total_reward = 0.0
        next_obs = torch.tensor(env.reset(), dtype=torch.float32).to(device)
        next_done = torch.zeros(1, dtype=torch.float32).to(device)
        cnt_done = 1
        if config['annealing']:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config['lr']
            agent.optimizer.param_groups[0]['lr'] = lrnow
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
            if done:
                cnt_done += 1
            with torch.no_grad():
                next_value = agent.get_value(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(rewards).to(device)
                lastgaelam = 0
                for t in reversed(range(config['num_steps'])):
                    if t == config['num_steps'] - 1:
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:
                        nextnonterminal = 1.0 - dones[t + 1]
                        nextvalues = values[t + 1]
                    delta = rewards[t] + config['gamma'] * nextvalues * nextnonterminal - values[t]
                    advantages[t] = lastgaelam = delta + config['gamma'] * config['gae_lambda'] * nextnonterminal * lastgaelam
                returns = advantages + values
        average_rewards.append(total_reward / cnt_done)
        average_return.append(returns)
        #================ Logging ===============================
        writer.add_scalar("total reward", total_reward, global_step)
        writer.add_scalar("average reward", np.mean(average_rewards[-100:]), global_step)
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
        if update % config['EVAL_ITR'] == 0:
            state = env.reset()
            buf = list()
            eval_total_rewards = []
            eval_total_reward = 0.0
            for i in range(100):
                if config['render_mode'] == 'rgb_array':
                    buf.append(env.render())
                action, prob, entropy, value = agent.get_action_and_value(torch.tensor(state, dtype=torch.float32).view(1, -1))
                next_state, reward, done, _ = env.step(action.cpu().numpy())
                state = next_state
                eval_total_reward += reward
                if done:
                    buf.append(env.render())
                    eval_total_rewards.append(eval_total_reward)
                    eval_total_reward = 0.0
                    state = env.reset()
            print("====================== EVALUALTION ======================")
            print(f"{global_step}: Epoch [{update}/{num_updates} : reward [{np.mean(eval_total_rewards):.3f}] \t")
            if config['record_vid']:
                save_frames(buf, name="PPO_")
        #================ Logging ================ 
        writer.add_scalar("critic_loss", v_loss.item(), global_step)
        writer.add_scalar("actor_loss", pg_loss.item(), global_step)
        if update % config['EVAL_ITR'] == 0:
            print(f"[{global_step}] Mean Reward: {np.mean(average_rewards[-100:]):.3f}")
            print(f"[{global_step}] Mean Return: {np.mean(average_return[-100:]):.3f}")
            print(f"[{global_step}] Value Loss : {v_loss.item():.3f}")
            print(f"[{global_step}] Critic Loss : {pg_loss.item():.3f}")
        #================ Logging ================ 
    save_model(agent, config, name="PPO", type='ppo')
    env.close()
    writer.close()

if __name__ == "__main__":
    # TensorBoard writer setup
    writer = SummaryWriter(comment='-PPO')

    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="PPO_MAP_10_10_HOLES_10.yaml", help="Path to a training config in /data/config")
    args = parser.parse_args()

    # Load configuration from YAML file
    config = load_yaml(args.config)
    config['device'] = get_device()

    # Display loaded configuration
    print("============ CONFIG ================")
    print(config)

    # Set device for training
    device = get_device()

    # Initialize environment
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'],
                       potholes=config['potholes'], traffic_jams=config['traffic_jams'])

    # Initialize PPO agent
    agent = PPOAgent(env, config)

    # Train the agent
    train(agent, env, config, writer, device)