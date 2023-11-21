import numpy as np
import pygame
import gymnasium as gym
from ray.rllib.algorithms import ppo
from gymnasium import spaces

from RL_Pizza_Delivery.visual.assets import COLOR, OBJECTS
from RL_Pizza_Delivery.env.rewards import REWARDS

class ENV_OBSTACLE(gym.Env):
    def __init__(self, config):
        super(ENV_OBSTACLE, self).__init__()
        self.window_size = 512
        # self.size = config['map_size'][0]
        self.size = 5
        # self.render_mode = config['render_mode']
        self.render_mode = None
        # self.map_size = config['map_size']
        self.map_size = (5, 5)
        self._max_times = 10 * self.size
        self.current_pos = None
        self.goal_pos = None
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, self.map_size[0] - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, self.map_size[0] - 1, shape=(2,), dtype=int),
            }
        )
        self.window = None
        self.clock = None
        # self.num_potholes = config['potholes']
        self.num_potholes = 0
        # self.num_traffic_jams = config['traffic_jams']
        self.num_traffic_jams = 0
        self.holes = np.zeros((self.num_potholes, 2))
        self.traffic_jams = np.zeros((self.num_traffic_jams, 2))
        # self.seed = config['seed']
        self.seed = 0
        self._timestep = 0
        self.reset()

    @property
    def observation_dim(self):
        return self.map_size[0] * self.map_size[1]
    
    @property
    def action_dim(self):
        return 4

    def reset(self, seed=1, options=None):
        self._timestep = 0
        self._holes, self._traffic_jams = set(), set()
        _i = 0
        self.current_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.goal_pos = self.current_pos
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

        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.window_size, self.window_size))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        self.canvas = pygame.Surface((self.window_size, self.window_size))
        self.canvas.fill((255, 255, 255))
        self.pix_square_size = (
            self.window_size / self.size
        )  # The size of a single grid square in pixels

        # First we draw the target
        pygame.draw.rect(
            self.canvas,
            (255, 0, 0),
            pygame.Rect(
                self.pix_square_size * self.goal_pos,
                (self.pix_square_size, self.pix_square_size),
            ),
        )
        # Now we draw the agent
        pygame.draw.circle(
            self.canvas,
            (0, 0, 255),
            (self.current_pos + 0.5) * self.pix_square_size,
            self.pix_square_size / 3,
        )

        # Finally, add some gridlines
        for x in range(self.size + 1):
            pygame.draw.line(
                self.canvas,
                0,
                (0, self.pix_square_size * x),
                (self.window_size, self.pix_square_size * x),
                width=3,
            )
            pygame.draw.line(
                self.canvas,
                0,
                (self.pix_square_size * x, 0),
                (self.pix_square_size * x, self.window_size),
                width=3,
            )

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
        #==============REWARDS================================#
        if self.render_mode == 'human':
            self._render_frame()
        return self._get_observation(), reward, done, {}, {}

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

    def render(self, mode='human'):
        pass

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    import ray
    from ray import tune
    from ray.rllib.env import MultiAgentEnv

    # Test Custom Enviroment
    config = {"epochs": 10000, "lr": 1e-3, 'potholes': 0, 'traffic_jams': 0
            , 'map_size': (5, 5), 'render_mode': None, 'epoch_print_cnt': 500
            , 'seed' : 1}

    # Register the environment with Ray
    ray.init()

    algo = ppo.PPO(env=ENV_OBSTACLE, config={
    "config": config,  # config to pass to env class
    })
    while True:
        print(algo.train())
