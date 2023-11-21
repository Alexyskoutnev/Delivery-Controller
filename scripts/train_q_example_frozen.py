from RL_Pizza_Delivery.algo.qlearning import QAgent

import numpy as np
import os
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import torch
import imageio

from queue import Queue
import statistics

def eval_traj(env, agent, epoch):
    frames = []
    state = env.reset()[0]
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32).view(1, -1))
        next_state, reward, done, _, _ = env.step(action)
        agent.update(
            torch.tensor(state, dtype=torch.float32).view(1, -1),
            action,
            reward,
            torch.tensor(next_state, dtype=torch.float32).view(1, -1),
            done
        )
        state = next_state
        total_reward += reward
        frames.append(env.render())
    print(f"Eval: Total reward: {total_reward}")
    path = os.path.join('../data/video', 'lunar_lander_animation_' + str(epoch) + '.gif')
    imageio.mimsave(path, frames)

# Q-learning implementation
def q_learning(env, agent, config=None):

    def average(rewards):
        return np.mean(reward)
    
    queue = Queue()

    for epoch in range(config['epochs']):
        state = env.reset()[0]
        done = False
        total_reward = 0
        
        while not done:
            action = agent.select_action(torch.tensor(state, dtype=torch.float32).view(1, -1))
            next_state, reward, done, _, _ = env.step(action)
            agent.update(
                torch.tensor(state, dtype=torch.float32).view(1, -1),
                action,
                reward,
                torch.tensor(next_state, dtype=torch.float32).view(1, -1),
                done
            )
            state = next_state
            total_reward += reward

        queue.put(total_reward)
        if epoch % config['epoch_print_cnt'] == 0 and epoch != 0:
            print(f"Episode {epoch}, Total reward: {total_reward}")
            eval_traj(env, agent, epoch)
        if queue.qsize() >= 100:
            mean_value = statistics.mean(queue.queue)
            queue.get()
            print(f"Mean values [{mean_value}]")

if __name__ == "__main__":
    config = {"epochs": 50000, "lr": 1e-2, 'potholes': 0, 'traffic_jams': 0
            , 'map_size': (5, 5), 'render_mode': 'rgb_array', 'epoch_print_cnt': 500}
    gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=True)
    desc=["SFFF", "FHFH", "FFFH", "HFFG"]
    env = gym.make('FrozenLake-v1', desc=generate_random_map(size=8))
    input_size, output_size = env.observation_space.shape[0],  env.action_space.n
    agent = QAgent(input_size, output_size)
    q_learning(env, agent, config=config)