import torch
import argparse

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, SAVE_DIR, load_yaml


def load(model_path, config):
    model = QAgent(env, buffer, lr=config['LR'], gamma=config['GAMMA'])
    state_dict = torch.load(model_path, map_location=torch.device('cpu'))
    model.net.load_state_dict(state_dict)
    return model

class Controller(object):
    def __init__(self, model_path, config) -> None:
        self.model = load(model_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="../data/models/DQN-2023-11-27 13:49:59.521207.pt", help="Path to a model in /data/models")
    args = parser.parse_args()
    # config = {"map_size": (5, 5), "potholes" : 0, "traffic_jams": 0, "render_mode": 'human',
    #           "GAMMA" : 0.99, "BATCH_SIZE": 32, "REPLAY_SIZE": 10000, "LR": 1e-4, "SYNC_TARGET_FRAMES": 1000,
    #           "REPLAY_START_SIZE": 10000, "EPSILON_DECAY_LAST_FRAME": 100000, "EPSILON_START": 1.0, "EPSILON_FINAL": 0.01,
    #           "epochs" : 100000, "EVAL_ITR": 10000, "record_vid" : True, "print_itr": 100, "eval_steps" : 100}
    config = load_yaml()
    controller = Controller(args.model, config)
    eval(agent, env, buffer, config)