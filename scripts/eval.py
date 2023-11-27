import torch
import argparse

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model

def eval(agent, env, buffer, config=None):
    state = env.reset()
    buf = list()
    eval_total_rewards = 0
    for i in range(config['eval_steps']):
        if config['render_mode'] == 'rgb_array':
            buf.append(env.render())
        action = agent.select_action(torch.tensor(state, dtype=torch.float32).view(1, -1), espilon=0)
        next_state, reward, done, _ = env.step(action)
        state = next_state
        eval_total_rewards += reward
        print_action(action)
        if done:
            print(f"Test reward: [{eval_total_rewards:.3f}] \t")
            eval_total_rewards = 0
            buf.append(env.render())
            state = env.reset()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="../data/models/DQN-2023-11-27 13:49:59.521207.pt", help="Path to a model in /data/models")
    args = parser.parse_args()
    config = {"map_size": (5, 5), "potholes" : 0, "traffic_jams": 0, "render_mode": 'human',
              "GAMMA" : 0.99, "BATCH_SIZE": 32, "REPLAY_SIZE": 10000, "LR": 1e-4, "SYNC_TARGET_FRAMES": 1000,
              "REPLAY_START_SIZE": 10000, "EPSILON_DECAY_LAST_FRAME": 100000, "EPSILON_START": 1.0, "EPSILON_FINAL": 0.01,
              "epochs" : 100000, "EVAL_ITR": 10000, "record_vid" : True, "print_itr": 100, "eval_steps" : 100}
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    buffer = ExperienceBuffer(config['REPLAY_SIZE'])
    agent = QAgent(env, buffer, lr=config['LR'], gamma=config['GAMMA'])
    load_model(args.model, agent)
    eval(agent, env, buffer, config)