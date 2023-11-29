import torch
import argparse

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.algo.ppo import PPOAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, load_yaml

def eval(agent, env, config=None, type='dqn'):
    state = env.reset()
    buf = list()
    eval_total_rewards = 0
    for i in range(config['eval_steps']):
        if config['render_mode'] == 'rgb_array':
            buf.append(env.render())
        if config['render_mode'] == 'human':
            env.render()
        if type == 'dqn':
            action = agent.select_action(torch.tensor(state, dtype=torch.float32).view(1, -1), espilon=0.1)
        elif type == 'ppo':
            action, prob, entropy, value = agent.get_action_and_value(torch.tensor(state, dtype=torch.float32).view(1, -1))
            action = action.item()
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
    parser.add_argument("-c", "--config", type=str, default="DQN_MAP_5_5_HOLES_0.yaml", help="Path to a training config in /data/config")
    parser.add_argument("-a", "--agent", type=str, default="dqn", help="Path to a training config in /data/config")
    args = parser.parse_args()
    agent_type = str(args.agent)
    config = load_yaml(args.config)
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    if agent_type == 'dqn':
        buffer = ExperienceBuffer(config['REPLAY_SIZE'])
        type = 'dqn'
        agent = QAgent(env, buffer, lr=config['LR'], gamma=config['GAMMA'])
        load_model(args.model, agent, type='dqn')
    elif agent_type == 'ppo':
        agent = PPOAgent(env, config)
        type = 'ppo'
        load_model(args.model, agent, type='ppo')
    eval(agent, env, config, type)