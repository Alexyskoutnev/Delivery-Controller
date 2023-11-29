import torch
import argparse
import os

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, SAVE_DIR, load_yaml

MODEL_DIR = "./data/models/"

def load(env, path, model):
    path = os.path.join(MODEL_DIR, path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.net.load_state_dict(state_dict)

class Controller(object):
    def __init__(self, env, model_path, properties=None) -> None:
        agent = QAgent(env)
        load(env, model_path, agent)
        self.model = agent.net
    
    def forward(self):
        pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="DQN_potholes-2_5_5.pt", help="Path to a model in /data/models")
    parser.add_argument("-c", "--config", type=str, default="DQN_MAP_5_5_HOLES_2.yaml", help="Path to a training config in /data/config")
    args = parser.parse_args()
    config = load_yaml(args.config)
    env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
    buffer = ExperienceBuffer(config['REPLAY_SIZE'])
    controller = Controller(env, args.model)
