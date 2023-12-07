import torch
import argparse
import os
import numpy as np
import heapq

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.controller.monitor import Monitor
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.algo.ppo import PPOAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, SAVE_DIR, load_yaml

MODEL_DIR = "./data/final_models/"

def load(env, path, model):
    path = os.path.join(MODEL_DIR, path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.net.load_state_dict(state_dict)

class Controller(object):
    def __init__(self, env, model, properties=None, monitor_flag=False) -> None:
        """
        Controller class for controlling an agent using a specified model.

        Parameters:
        - env: The environment.
        - model: The model used for control.
        - properties: Optional dictionary specifying properties to monitor.
        - monitor_flag: Flag to enable or disable monitoring.
        """
        self.model = model
        self.env = env
        self.init_state = env.reset()
        self.property = properties
        self.map = env.get_map
        self.map_size = env.map_size
        self.current_pos = env.get_pose
        self.goal_pos = env.get_goal
        self.max_itr = 100
        self.monitor = Monitor(self.current_pos, properties, self.map) if monitor_flag else None 
        self.path = []

    def run(self):
        """Run the controller to solve the task."""
        self._solve()

    def _solve(self):
        """Internal method to solve the task."""
        _itr = 0
        state = self.init_state
        while (_itr < self.max_itr):
            action = self.model(torch.tensor(state, dtype=torch.float32)).item()
            property_bool = self._test(action)
            if property_bool:
                action = self._safe_search()
                if action == None:
                    break
            state = self._step(action)
            if self.monitor:
                self.monitor.update(self.current_pos)
            if np.array_equal(self.current_pos, self.goal_pos):
                print("AT GOAL")
                break
            
    def _step(self, action):
        """
        Perform an action in the environment and update the current position.

        Parameters:
        - action: The action to perform.

        Returns:
        - state: The new state after performing the action.
        """
        state, _, _, _ = self.env.step(action)
        self.current_pos = self.env.current_pos
        return state
        
    def _hole(self, state):
        """Check if the current state is a hole."""
        return True if self.map[state[0]][state[1]] == 1. else False

    def _test(self, action):
        """
        Test if the current action leads to a violation of a specified property.

        Parameters:
        - action: The action to test.

        Returns:
        - bool: True if a violation is detected, False otherwise.
        """

        if self.property['POTHOLES']:
            next_state = self._lookahead(action, update=False)
            inhole = self._hole(next_state)
            if inhole:
                return True
            else: 
                return False
        return False
    
    def _compute_action(self, prev_state, state):
        """
        Compute the action needed to transition from a previous state to the current state.

        Parameters:
        - prev_state: The previous state.
        - state: The current state.

        Returns:
        - action: The computed action.
        """
        prev_state = np.array(prev_state)
        state = np.array(state)
        diff = state - prev_state
        idx = np.argmax((diff == 1) | (diff == -1))
        if idx == 0: #if diffence is along y-axis
            if diff[idx] == 1: #down
                action = 1 #right command
            elif diff[idx] == -1:
                action = 0 #left command
        elif idx == 1: #if difference is along x-axis
            if diff[idx] == 1:
                action = 3 #down command
            elif diff[idx] == -1:
                action = 2 #up command
        return action

    def _safe_search(self):
        """
        Perform a safe search using A* algorithm to find a safe path.

        Returns:
        - action: The computed action for the safe path.
        """
        path = self.astar(self.map.tolist(), tuple(self.current_pos), tuple(self.goal_pos))
        if len(path) >= 2:
            next_state = path[1]
            action = self._compute_action(path[0], path[1])
        else:
            print("FAIL TO FIND A PATH")
            return None
        return action

    def astar(self, grid, start, goal):
        """
        A* algorithm for pathfinding on a 2D grid map.

        Parameters:
        - grid (list of lists): 2D grid map where 0 represents free space, and 1 represents obstacles.
        - start (tuple): Starting coordinates (x, y).
        - goal (tuple): Goal coordinates (x, y).

        Returns:
        - list of tuples: The path from start to goal, including both start and goal.
        """

        def heuristic(node, goal):
            return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

        rows, cols = len(grid), len(grid[0])
        open_set = []  # Priority queue for nodes to be evaluated
        closed_set = set()  # Set of nodes already evaluated

        heapq.heappush(open_set, (0, start, []))  # Initial node: (f_cost, node, path)

        while open_set:
            f_cost, current, path = heapq.heappop(open_set)
            if current == goal:
                return path + [current]
            if current in closed_set:
                continue
            closed_set.add(current)
            for neighbor in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                x, y = current[0] + neighbor[0], current[1] + neighbor[1]
                if 0 <= x < rows and 0 <= y < cols and grid[x][y] == 0.:
                    neighbor_node = (x, y)
                    new_path = path + [current]
                    g_cost = len(new_path)
                    h_cost = heuristic(neighbor_node, goal)
                    f_cost = g_cost + h_cost
                    heapq.heappush(open_set, (f_cost, neighbor_node, new_path))
        return []  # No path found

    def _lookahead(self, action, update=False):
        """
        Compute the next position based on the given action.

        Parameters:
        - action: The action to be taken.

        Returns:
        - current_pos: The computed next position.
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

    def _state(self):
        """
        Get the current state representation.

        Returns:
        - torch.tensor: The tensor representing the current state.
        """
        agent_obs = np.array([self.current_pos[0] * self.map_size[0] + self.current_pos[1] * self.map_size[1]], dtype=np.float32)
        goal_obs = np.array([self.goal_pos[0] * self.map_size[0] + self.goal_pos[1] * self.map_size[1]], dtype=np.float32)
        map_obs = np.transpose(self.map).flatten()
        obs = np.concatenate((agent_obs, goal_obs, map_obs))
        return torch.tensor(obs, dtype=torch.float32)
