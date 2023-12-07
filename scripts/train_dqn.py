import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import time
from tensorboardX import SummaryWriter

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.visual.assets import COLOR, OBJECTS
from RL_Pizza_Delivery.env.rewards import REWARDS
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, load_yaml

def eval(agent : QAgent, env : ENV_OBSTACLE, config : dict, frameid : int):
    """
    Evaluate the specified Q-learning agent in the given environment.

    Parameters:
    - agent (QAgent): The Q-learning agent to be evaluated.
    - env (ENV_OBSTACLE): The environment in which the agent is evaluated.
    - config (dict): Configuration parameters for evaluation.
    - frameid (int): Identifier for the evaluation frame or iteration.

    Returns:
    None
    """
    buf = []
    eval_total_rewards = 0.0
    eval_rewards = []
    state = env.reset()
    for i in range(config['eval_steps']):
        if config['render_mode'] == 'rgb_array':
            buf.append(env.render())
        elif config['render_mode'] == 'human':
            env.render()
        action = agent.select_action(torch.tensor(state, dtype=torch.float32), 0)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        eval_total_rewards += reward
        if done:
            state = env.reset()
            eval_rewards.append(eval_total_rewards)
            eval_total_rewards = 0.0
            buf.append(env.render())
    print("====================== EVALUALTION ======================")
    print(f"{frameid}: reward [{np.mean(eval_rewards):.3f}] \t")
    if config['record_vid']:
        save_frames(buf, name="DQN_")

def train(agent : QAgent, env : ENV_OBSTACLE, buffer : ExperienceBuffer, writer : SummaryWriter = None, config : dict = None):
    """
    Train the specified Q-learning agent using experience replay.

    Parameters:
    - agent (QAgent): The Q-learning agent to be trained.
    - env (ENV_OBSTACLE): The environment in which the agent is trained.
    - buffer (ExperienceBuffer): The experience replay buffer.
    - writer (SummaryWriter, optional): TensorBoard SummaryWriter for logging. Defaults to None.
    - config (dict, optional): Configuration parameters for training. Defaults to None.

    Returns:
    None
    """
    epsilon = config['EPSILON_START']
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    while (frame_idx < config['total_timesteps']):
        #================ Experience Collection ================ 
        frame_idx += 1
        epsilon = max(config['EPSILON_FINAL'], config['EPSILON_START'] -
                        frame_idx / config['EPSILON_DECAY_LAST_FRAME'])
        reward = agent.play_step(epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)
        #================ Experience Collection ================ 
        #================ Backprob Step ========================
        if len(buffer) < config['REPLAY_START_SIZE']:
            continue
        if frame_idx % config['SYNC_TARGET_FRAMES'] == 0:
            agent.net_target.load_state_dict(agent.net.state_dict())
        batch = buffer.sample(config['BATCH_SIZE'])
        loss = agent.update(batch)
        writer.add_scalar("loss", loss.item(), frame_idx)
        #================ Backprob Update ======================        
        #================ Logging Update =======================
        if frame_idx % config['print_itr'] == 0:
                print("%d: done %d games, reward %.3f, "
                    "eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), m_reward, epsilon,
                    speed
                ))
        
        writer.add_scalar("loss", loss, frame_idx)
        #================ Logging Step ================ 
        #================ EVAL STEP ===================
        if frame_idx % config['EVAL_ITR'] == 0:
            eval(agent, env, config, frame_idx)
        #================ EVAL STEP ===================
    #================= Save/Close =====================
    save_model(agent.net, config, "DQN")
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="DQN_MAP_5_5_HOLES_0.yaml", help="Path to a training config in /data/config")
    args = parser.parse_args()
    config = load_yaml(args.config)
    config['EPSILON_DECAY_LAST_FRAME'] = int(config['total_timesteps'] * 0.9)
    print("================== CONFIG =======================")
    print(config)
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    buffer = ExperienceBuffer(config['REPLAY_SIZE'])
    agent = QAgent(env, buffer, lr=config['LR'], gamma=config['GAMMA'])
    writer = SummaryWriter(comment='-DQN')
    train(agent, env, buffer, writer, config)