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
        return self.map_size[0] * self.map_size[1]
    
    @property
    def action_dim(self):
        return 4

    def reset(self):
        self._timestep = 0
        self._holes, self._traffic_jams = set(), set()
        _i = 0
        self.current_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        # self.current_pos = np.array([0, 0], dtype=int)
        self.goal_pos = self.current_pos
        # self.goal_pos = np.array([self.map_size[0] - 1, self.map_size[0] - 1])
        while np.array_equal(self.goal_pos, self.current_pos):
            self.goal_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        for i, pothole in enumerate(range(self.num_potholes)):
            hole = self.np_random.integers(0, self.size, size=2, dtype=int)
            hole_tuple = tuple(hole)
            while (np.array_equal(self.current_pos, hole) or np.array_equal(self.goal_pos, hole) or hole_tuple in self._holes) and _i < 1000:
                hole = self.np_random.integers(0, self.size, size=2, dtype=int)
                _i += 1
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
        goal_reward = REWARDS.AT_GOAL if done and self._timestep < self._max_times else 0.0
        # reward = REWARDS.SCALE_DIST*(1 / (np.sqrt((self.current_pos[0] - self.goal_pos[0])**2) + 0.0001) + (self.current_pos[1] - self.goal_pos[1])**2) + pothole_reward + goal_reward
        reward = -REWARDS.SCALE_DIST*(np.sqrt((self.current_pos[0] - self.goal_pos[0])**2) + (self.current_pos[1] - self.goal_pos[1])**2) + pothole_reward + goal_reward
        # reward = -0.1 + goal_reward + pothole_reward
        #==============REWARDS================================#
        if self.render_mode == 'human':
            self._render_frame()
        return self._get_observation(), reward, done, {}

    def _get_observation(self):
        """
        Get the current observation of the environment.

        Returns:
            np.ndarray: The observation array representing the current state.
        """
        observation = np.zeros(self.map_size)
        observation[self.current_pos[0], self.current_pos[1]] = OBJECTS.CURRENT
        observation[self.goal_pos[0], self.goal_pos[1]] = OBJECTS.GOAL
        for hole in self._holes:
            if hole == tuple(self.current_pos):
                observation[hole[0], hole[1]] = OBJECTS.POTHOLE_N_CURRENT
            else:
                observation[hole[0], hole[1]] = OBJECTS.POTHOLE
        for traffic_jam in self._traffic_jams:
            if traffic_jam in tuple(self.current_pos):
                observation[traffic_jam[0], traffic_jam[1]] = OBJECTS.TRAFFIC_JAM_N_CURRENT
            else:
                observation[hole[0], hole[1]] = OBJECTS.TRAFFIC_JAM
        observation = np.transpose(observation)
        return observation.flatten()

if __name__ == "__main__":
     # Training parameters
    potholes, traffic_jams = 0, 0
    map_size = (5, 5)
    render_mode = 'None'
    # render_mode = 'human'
    grid_size = 5
    state_size = grid_size * grid_size
    action_size = 4
    learning_rate = 0.0001
    gamma = 0.99
    epsilon_decay = 0.95
    epsilon_min = 0.01
    episodes = 1000
    max_moves = grid_size * 10
    GAMMA = 0.99
    BATCH_SIZE = 32
    REPLAY_SIZE = 10000
    LEARNING_RATE = 1e-4
    SYNC_TARGET_FRAMES = 1000
    REPLAY_START_SIZE = 10000

    EPSILON_DECAY_LAST_FRAME = 100000
    EPSILON_START = 1.0
    EPSILON_FINAL = 0.01

    env = ENV_OBSTACLE(map_size= map_size, render_mode=render_mode, potholes=potholes, traffic_jams=traffic_jams)
    input_size, output_size = env.observation_dim, env.action_dim
    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = QAgent(env, buffer, lr=learning_rate, gamma=gamma)
    epsilon = EPSILON_START
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    writer = SummaryWriter(comment="-")

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START -
                        frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(epsilon) 

        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print("%d: done %d games, reward %.3f, "
                  "eps %.2f, speed %.2f f/s" % (
                frame_idx, len(total_rewards), m_reward, epsilon,
                speed
            ))
            writer.add_scalar("epsilon", epsilon, frame_idx)
            writer.add_scalar("speed", speed, frame_idx)
            writer.add_scalar("reward_100", m_reward, frame_idx)
            writer.add_scalar("reward", reward, frame_idx)

        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            agent.net_target.load_state_dict(agent.net.state_dict())

        batch = buffer.sample(BATCH_SIZE)
        loss = agent.update(batch)
        writer.add_scalar("loss", loss, frame_idx)