import torch
import argparse
import os
import numpy as np
import heapq

from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent
from RL_Pizza_Delivery.algo.ppo import PPOAgent
from RL_Pizza_Delivery.utils.torch_utils import save_model, save_frames, print_action, load_model, SAVE_DIR, load_yaml

MODEL_DIR = "./data/final_models/"
PROPERTY_CONFIG = "controller.yaml"

def load(env, path, model):
    path = os.path.join(MODEL_DIR, path)
    state_dict = torch.load(path, map_location=torch.device('cpu'))
    model.net.load_state_dict(state_dict)

class Controller(object):
    def __init__(self, env, model, properties=None) -> None:
        self.model = model
        self.env = env
        self.property = properties
        self.map = env.get_map
        self.map_size = env.map_size
        self.current_pos = env.get_pose
        self.goal_pos = env.get_goal
        self.max_itr = 100
        self.path = []
        self._solve()

    def _solve(self):
        _itr = 0
        while (_itr < self.max_itr):
            # print(f"PREVIOUS STATE {self.current_pos}")
            state = self._state()
            action = self.model(state)
            property_bool = self._test(action)
            if property_bool:
                action = self._safe_search(model_action=action)
                next_state = self._step(action, update=True)
            else:
                next_state = self._step(action, update=True)
            property_bool = self._test(action)
            print(f"Property_bool {property_bool}")
            # print(f"Current_State {next_state}")
            self.path.append(next_state)
            if np.array_equal(next_state, self.goal_pos):
                print("AT GOAL")
                return
        
    def _hole(self, state):
        return True if self.map[state[0]][state[1]] == 1. else False

    def _test(self, action):
        if self.property['POTHOLES']:
            next_state = self._step(action, update=False)
            inhole = self._hole(next_state)
            if inhole:
                return True
            else: 
                return False
        return False
    
    def _compute_action(self, prev_state, state):
        prev_state = np.array(prev_state)
        state = np.array(state)
        diff = state - prev_state
        idx = np.argmax((diff == 1) | (diff == -1))
        if idx == 0: #if diffence is along y-axis
            if diff[idx] == 1: #down
                action = 3 #down command
            elif diff[idx] == -1:
                action = 2 #up command
        elif idx == 1: #if difference is along x-axis
            if diff[idx] == 1:
                action = 1 #right command
            elif diff[idx] == -1:
                action = 0 #left command
        return action

    def _safe_search(self, model_action=None):
        path = self.astar(self.map.tolist(), tuple(self.current_pos), tuple(self.goal_pos))
        if len(path) >= 2:
            print("USED A STAR STEP")
            next_state = path[1]
            action = self._compute_action(path[0], path[1])
        else:
            action = model_action
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

    def _step(self, action, update=False):
        if action == 0: #UP
            current_pos = max(0, self.current_pos[0] - 1), self.current_pos[1]
        elif action == 1: #Down
            current_pos = min(self.map_size[0] -1, self.current_pos[0] + 1), self.current_pos[1]
        elif action == 2: #left
            current_pos = self.current_pos[0], max(0, self.current_pos[1] - 1)
        elif action == 3:
            current_pos = self.current_pos[0], min(self.map_size[1] - 1, self.current_pos[1] + 1)
        if update:
            self.current_pos = np.array(current_pos, dtype=int)
            return self.current_pos
        else:
            return current_pos

    def _convert_to_coords(self, state):
        i, j = state[0:2]

    def _state(self):
        agent_obs = np.array([self.current_pos[0] * self.map_size[0] + self.current_pos[1] * self.map_size[1]], dtype=np.float32)
        goal_obs = np.array([self.goal_pos[0] * self.map_size[0] + self.goal_pos[1] * self.map_size[1]], dtype=np.float32)
        map_obs = np.transpose(self.map).flatten()
        obs = np.concatenate((agent_obs, goal_obs, map_obs))
        return torch.tensor(obs, dtype=torch.float32)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="./data/final_models/PPO_potholes-10_10_10.pt", help="Path to a model in /data/models")
    parser.add_argument("-c", "--config", type=str, default="PPO_MAP_10_10_HOLES_10.yaml", help="Path to a training config in /data/config")
    parser.add_argument("-a", "--agent", type=str, default="ppo", help="Path to a training config in /data/config")
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
        model = PPOAgent(env, config)
        type = 'ppo'
        load_model(args.model, model, type='ppo')
        properties = load_yaml(PROPERTY_CONFIG)
    
    controller = Controller(env, model, properties)
    
    
