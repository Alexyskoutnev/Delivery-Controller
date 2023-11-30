import torch
import argparse

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.controller.controller import Controller
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, SAVE_DIR, load_yaml

class Monitor(object):
    def __init__(self, init_state, property, map):
        self.violations = 0
        self.property = property
        self.step = 0
        self.map = map
        self.map_size = (len(map), len(map[0]))
        breakpoint()
        self.update(init_state)

    def _hole(self, state):
        return True if self.map[state[0]][state[1]] == 1. else False

    def print(self):
        for property in self.property.keys():
            print(f"[{self.step}]  {property} VIOLATIONS | {self.violations}")

    def _test(self, state):
        if self.property['POTHOLES']:
            inhole = self._hole(state)
            if inhole:
                return True
        return False

    def update(self, state):
        property_bool = self._test(state)
        if property_bool:
            self.violations += 1
        self.step += 1
        self.print()
    
    def _lookahead(self, action):
        if action == 0: #UP
            current_pos = max(0, self.current_pos[0] - 1), self.current_pos[1]
        elif action == 1: #Down
            current_pos = min(self.map_size[0] -1, self.current_pos[0] + 1), self.current_pos[1]
        elif action == 2: #left
            current_pos = self.current_pos[0], max(0, self.current_pos[1] - 1)
        elif action == 3:
            current_pos = self.current_pos[0], min(self.map_size[1] - 1, self.current_pos[1] + 1)
        return current_pos

