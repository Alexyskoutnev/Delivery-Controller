import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tensorboardX import SummaryWriter

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.visual.assets import COLOR, OBJECTS
from RL_Pizza_Delivery.env.rewards import REWARDS
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames


def train(agent, env, buffer, writer=None, config=None):
    epsilon = config['EPSILON_START']
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    for epoch in range(config['epochs']):
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
        #================ Backprob Update ======================        
        #================ Logging Update =======================
        if epoch % config['print_itr'] == 0:
                print("%d: done %d games, reward %.3f, "
                    "eps %.2f, speed %.2f f/s" % (
                    frame_idx, len(total_rewards), m_reward, epsilon,
                    speed
                ))
        if epoch % config['EVAL_ITR'] == 0:
            state = env.reset()
            buf = list()
            eval_total_rewards = 0
            for i in range(100):
                if config['render_mode'] == 'rgb_array':
                    buf.append(env.render())
                action = agent.select_action(torch.tensor(state, dtype=torch.float32).view(1, -1), espilon=0)
                next_state, reward, done, _ = env.step(action)
                state = next_state
                eval_total_rewards += reward
                if done:
                    buf.append(env.render())
                    break
            print("====================== EVALUALTION ======================")
            print(f"{frame_idx}: Epoch [{epoch}/{config['epochs']}] : reward [{eval_total_rewards:.3f}] \t")
            if config['record_vid']:
                save_frames(buf)
        writer.add_scalar("loss", loss, frame_idx)
        #================ Logging Step ================ 
    #================= Save/Close =====================
    save_model(agent.net)
    writer.close()

if __name__ == "__main__":
    config = {"map_size": (2, 2), "potholes" : 0, "traffic_jams": 0, "render_mode": 'rgb_array',
              "GAMMA" : 0.99, "BATCH_SIZE": 32, "REPLAY_SIZE": 10000, "LR": 1e-4, "SYNC_TARGET_FRAMES": 1000,
              "REPLAY_START_SIZE": 10000, "EPSILON_DECAY_LAST_FRAME": 100000, "EPSILON_START": 1.0, "EPSILON_FINAL": 0.01,
              "epochs" : 100000, "EVAL_ITR": 10000, "record_vid" : True, "print_itr": 1000}
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    buffer = ExperienceBuffer(config['REPLAY_SIZE'])
    agent = QAgent(env, buffer, lr=config['LR'], gamma=config['GAMMA'])
    writer = SummaryWriter(comment='-DQN')
    train(agent, env, buffer, writer, config)