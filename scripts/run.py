import argparse
import numpy as np


from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.algo.ppo import PPOAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, SAVE_DIR, load_yaml
from RL_Pizza_Delivery.controller.controller import Controller

PROPERTY_CONFIG = "controller.yaml"

def main(controller):
    """
    Main function to run the reinforcement learning agent using the provided controller.

    Args:
        controller (Controller): The controller object managing the agent-environment interaction.
    """
    controller.run()
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="./data/final_models/PPO_potholes-10_10_10.pt", help="Path to a model in /data/models")
    parser.add_argument("-c", "--config", type=str, default="run.yaml", help="Path to a training config in /data/config")
    parser.add_argument("-a", "--agent", type=str, default="ppo", help="Path to a training config in /data/config")
    parser.add_argument('-s', action='store_true', help='When present, runtime monitor is implemented')
    args = parser.parse_args()
    agent_type = str(args.agent)
    config = load_yaml(args.config)
    gent_type = str(args.agent)
    config = load_yaml(args.config)
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    if agent_type == 'dqn':
        buffer = ExperienceBuffer(config['REPLAY_SIZE'])
        type = 'dqn'
        agent = QAgent(env, buffer, lr=config['LR'], gamma=config['GAMMA'])
        load_model(args.model, agent, type='dqn')
    elif agent_type == 'ppo':
        model = PPOAgent(env, config)
        type = 'ppo'
        load_model(args.model, model, type='ppo')
        properties = load_yaml(PROPERTY_CONFIG)
    monitor_flag = True if args.s else False
    controller = Controller(env, model, properties, monitor_flag=monitor_flag)
    main(controller)
