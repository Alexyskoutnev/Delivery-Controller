import torch
import argparse

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, SAVE_DIR, load_yaml

<<<<<<< HEAD

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
=======
class Monitor(object):
    def __init__(self, init_state, property, map):
        """
        Monitor class to track violations of certain properties in the environment.

        Parameters:
        - init_state: The initial state of the environment.
        - property: Dictionary specifying the properties to monitor.
        - map: The map of the environment.
        """
        self.violations = 0
        self.property = property
        self.step = 0
        self.map = map
        self.map_size = (len(map), len(map[0]))
        self.update(init_state)

    def _hole(self, state):
        """Check if the current state is a hole."""
        return True if self.map[state[0]][state[1]] == 1. else False

    def print(self):
        """Print the violations for each property."""
        for property in self.property.keys():
            print(f"[{self.step}]  {property} VIOLATIONS | {self.violations}")

    def _test(self, state):
        """Test if the current state violates any property."""
        inhole = self._hole(state)
        if inhole:
            return True
        return False

    def update(self, state):
        """
        Update the monitor based on the current state.

        Parameters:
        - state: The current state of the environment.
        """
        property_bool = self._test(state)
        if property_bool:
            self.violations += 1
        self.step += 1
        self.print()
    
    def _lookahead(self, action):
        """
        Calculate the next position based on the current action.

        Parameters:
        - action: The action to be taken.

        Returns:
        - tuple: The next position.
        """
        if action == 0: #UP
            current_pos = max(0, self.current_pos[0] - 1), self.current_pos[1]
        elif action == 1: #Down
            current_pos = min(self.map_size[0] -1, self.current_pos[0] + 1), self.current_pos[1]
        elif action == 2: #Left
            current_pos = self.current_pos[0], max(0, self.current_pos[1] - 1)
        elif action == 3: #Right
            current_pos = self.current_pos[0], min(self.map_size[1] - 1, self.current_pos[1] + 1)
        return current_pos

>>>>>>> alexy-dev
