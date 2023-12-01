import numpy as np
import pygame
import torch
import torch.nn as nn
import torch.optim as optim
import time
from tensorboardX import SummaryWriter

from RL_Pizza_Delivery.env.env import ENV_BASE
from RL_Pizza_Delivery.visual.assets import COLOR, OBJECTS
from RL_Pizza_Delivery.env.rewards import REWARDS
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.qlearning import QAgent

class ENV_OBSTACLE(ENV_BASE):
    def __init__(self, potholes=0, traffic_jams=0, map_size=..., render_mode=None, seed=1):
        super().__init__(map_size, render_mode, seed, BASE_ENV_FLAG=False)
        self.num_potholes = potholes
        self.num_traffic_jams = traffic_jams
        self.holes = np.zeros((self.num_potholes, 2))
        self.traffic_jams = np.zeros((self.num_traffic_jams, 2))
        self.seed = seed
        self._timestep = 0
        self.reset()

    @property
    def observation_dim(self):
        return (self.map_size[0] * self.map_size[1]) + 2
    
    @property
    def action_dim(self):
        return 4

    def reset(self):
        self._timestep = 0
        self._holes, self._traffic_jams = set(), set()
        _i = 0
        self.current_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.goal_pos = np.array([self.map_size[0] - 1, self.map_size[0] - 1])
        while np.array_equal(self.goal_pos, self.current_pos):
            self.current_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        for i, pothole in enumerate(range(self.num_potholes)):
            hole = self.np_random.integers(0, self.size, size=2, dtype=int)
            hole_tuple = tuple(hole)
            while (np.array_equal(self.current_pos, hole) or np.array_equal(self.goal_pos, hole) or hole_tuple in self._holes) and _i < 1000:
                hole = self.np_random.integers(0, self.size, size=2, dtype=int)
                _i += 1
            hole_tuple = tuple(hole)
            self.holes[i] = hole
            self._holes.add(hole_tuple)
            _i = 0
        for i, pothole in enumerate(range(self.num_traffic_jams)):
            traffic_jam = self.np_random.integers(0, self.size, size=2, dtype=int)
            traffic_jam_tuple = tuple(traffic_jam)
            while (np.array_equal(self.current_pos, traffic_jam) or np.array_equal(self.goal_pos, traffic_jam) or traffic_jam_tuple in self._traffic_jams or traffic_jam_tuple in self._holes) and _i < 1000:
                traffic_jam = self.np_random.integers(0, self.size, size=2, dtype=int)
                _i += 1
            self.traffic_jams[i] = traffic_jam
            self._traffic_jams.add(traffic_jam_tuple)
            _i = 0

        if self.render_mode == 'human_mode':
            self._render_frame()

        return self._get_observation()

    def _render_frame(self):
        super()._render_frame()

        for idx in range(self.num_potholes):
            if np.array_equal(self.current_pos, self.holes[idx]):
                pygame.draw.circle(self.canvas, COLOR.gray, (self.holes[idx] + 0.5) * self.pix_square_size, self.pix_square_size / 3,)  # Draw a gray circle for the pothole
                pygame.draw.circle(self.canvas, COLOR.blue, (self.holes[idx] + 0.5) * self.pix_square_size, self.pix_square_size / 4,)  # Draw a Color.gray circle for the pothole
            else:
                pygame.draw.circle(self.canvas, COLOR.gray, (self.holes[idx] + 0.5) * self.pix_square_size, self.pix_square_size / 3,)  # Draw a Color.gray circle for the pothole
        
        for idx in range(self.num_traffic_jams):
            if np.array_equal(self.current_pos, self.holes[idx]):
                pygame.draw.rect(self.canvas, COLOR.black, pygame.Rect(
                self.pix_square_size * self.traffic_jams[idx] + 0.25,
                (self.pix_square_size / 1.5, self.pix_square_size / 1.5),),)  # Draw a gray circle for the pothole
                pygame.draw.circle(self.canvas, COLOR.blue, self.pix_square_size * self.traffic_jams[idx], self.pix_square_size / 4,)  # Draw a Color.gray circle for the pothole
            else:
                pygame.draw.rect(self.canvas, COLOR.black, pygame.Rect(
                self.pix_square_size * (self.traffic_jams[idx] + 0.25),
                (self.pix_square_size / 1.5, self.pix_square_size / 1.5),),)

        if self.render_mode == "human":
            self.window.blit(self.canvas, self.canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        elif self.render_mode == 'rgb_array':  # rgb_array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
                )

    def step(self, action):
        """
        Take a step in the environment.

        Args:
            action (int): The action to take.

        Returns:
            tuple: Tuple containing:
                - np.ndarray: The new observation after the step.
                - float: The reward obtained from the step.
                - bool: Whether the episode is finished.
                - dict: Additional information.
        """
        self._timestep += 1
        if action == 0: #UP
            self.current_pos = max(0, self.current_pos[0] - 1), self.current_pos[1]
        elif action == 1: #Down
            self.current_pos = min(self.map_size[0] -1, self.current_pos[0] + 1), self.current_pos[1]
        elif action == 2: #left
            self.current_pos = self.current_pos[0], max(0, self.current_pos[1] - 1)
        elif action == 3:
            self.current_pos = self.current_pos[0], min(self.map_size[1] - 1, self.current_pos[1] + 1)
        self.current_pos = np.array(self.current_pos, dtype=int)
        done = np.array_equal(self.current_pos, self.goal_pos) or self._max_times <= self._timestep
        #==============REWARDS================================#
        current_pos_tup = tuple(self.current_pos)
        pothole_reward = REWARDS.POTHOLE if current_pos_tup in self._holes else 0.0
        goal_reward = (self.map_size[0] * REWARDS.AT_GOAL) if done and self._timestep < self._max_times else 0.0
        reward = -REWARDS.SCALE_DIST*(np.sqrt((self.current_pos[0] - self.goal_pos[0])**2) + (self.current_pos[1] - self.goal_pos[1])**2) + pothole_reward + goal_reward
        # reward = -0.1 + goal_reward + pothole_reward
        #==============REWARDS================================#
        if self.render_mode == 'human':
            self._render_frame()
        return self._get_observation(), reward, done, {}

    @property
    def get_map(self):
        return self._render_observation

    @property
    def get_pose(self):
        return self.current_pos

    @property
    def get_goal(self):
        return self.goal_pos

    def _get_observation(self):
        """
        Get the current observation of the environment.

        Returns:
            np.ndarray: The observation array representing the current state.
        """
        observation = np.zeros(self.map_size)
        obstacle_obs = np.zeros(self.map_size)
        self._render_observation = np.zeros(self.map_size)
        agent_obs = np.array([self.current_pos[0] * self.map_size[0] + self.current_pos[1] * self.map_size[1]], dtype=np.float32)
        goal_obs = np.array([self.goal_pos[0] * self.map_size[0] + self.goal_pos[1] * self.map_size[1]], dtype=np.float32)
        for hole in self._holes:
            observation[hole[0], hole[1]] = 1
            self._render_observation[hole[0], hole[1]] = 1
        for traffic_jam in self._traffic_jams:
            if traffic_jam in tuple(self.current_pos):
                observation[traffic_jam[0], traffic_jam[1]] = OBJECTS.TRAFFIC_JAM_N_CURRENT
            else:
                observation[hole[0], hole[1]] = OBJECTS.TRAFFIC_JAM
        observation = np.transpose(observation).flatten()
        observation = np.concatenate((agent_obs, goal_obs, observation))
        return observation

if __name__ == "__main__":
    pass