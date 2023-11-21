import gymnasium as gym
from gymnasium import spaces
import pygame
import numpy as np

from RL_Pizza_Delivery.visual.assets import COLOR, OBJECTS

class ENV_BASE(gym.Env):
    """
    Custom environment for a 2D map navigation problem in Gym.

    Args:
        map_size (tuple): Size of the 2D map (rows, columns).
        seed (int): Seed for random number generation.

    Attributes:
        map_size (tuple): Size of the 2D map (rows, columns).
        current_pos (tuple): Current position of the agent on the map.
        goal_pos (tuple): Goal position that the agent needs to reach.
        seed (int): Seed for random number generation.
        action_space (gym.Space): Action space for the environment (Discrete).
        observation_space (gym.Space): Observation space for the environment (Box).
    """
    metadata = {"render_modes": ['human', 'rgb_array'], 'render_fps': 5}


    # def __init__(self, map_size=(10, 10), render_mode=None, seed=1, BASE_ENV_FLAG=True):
    def __init__(self, map_size=(10, 10), render_mode=None, seed=1, BASE_ENV_FLAG=True):
        """
        Initialize the environment.

        Args:
            map_size (tuple): Size of the 2D map (rows, columns).
            seed (int): Seed for random number generation.
        """
        super(ENV_BASE, self).__init__()
        self.window_size = 512
        self.size = map_size[0]
        self.render_mode = render_mode
        self.map_size = map_size
        self._max_times = 10 * self.size
        self.current_pos = None
        self.goal_pos = None
        self.seed = seed
        self.action_space = spaces.Discrete(4)
        self.BASE_ENV_FLAG = BASE_ENV_FLAG
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(0, map_size[0] - 1, shape=(2,), dtype=int),
                "target": spaces.Box(0, map_size[0] - 1, shape=(2,), dtype=int),
            }
        )
        self.window = None
        self.clock = None

    def reset(self):
        """
        Reset the environment to its initial state.

        Returns:
            np.ndarray: Initial observation of the environment.
        """
        super().reset(seed=self.seed)
        self.current_pos = self.np_random.integers(0, self.size, size=2, dtype=int)
        self.goal_pos = self.current_pos
        while np.array_equal(self.goal_pos, self.current_pos):
            self.goal_pos = self.np_random.integers(0, self.size, size=2, dtype=int)

        if self.render_mode == 'human_mode':
            self._render_frame()

        return self._get_observation()

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
        if action == 0: #UP
            self.current_pos = max(0, self.current_pos[0] - 1), self.current_pos[1]
        elif action == 1: #Down
            self.current_pos = min(self.map_size[0] -1, self.current_pos[0] + 1), self.current_pos[1]
        elif action == 2: #left
            self.current_pos = self.current_pos[0], max(0, self.current_pos[1] - 1)
        elif action == 3:
            self.current_pos = self.current_pos[0], min(self.map_size[1] - 1, self.current_pos[1] + 1)
        self.current_pos = np.array(self.current_pos, dtype=int)
        reward = -np.sqrt((self.current_pos[0] - self.goal_pos[0])**2 + (self.current_pos[1] - self.goal_pos[1])**2)
        done = np.array_equal(self.current_pos, self.goal_pos)
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
        return observation

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

        if self.BASE_ENV_FLAG:
            if self.render_mode == "human":
                # The following line copies our drawings from `self.canvas` to the visible window
                self.window.blit(self.canvas, self.canvas.get_rect())
                pygame.event.pump()
                pygame.display.update()

                # We need to ensure that human-rendering occurs at the predefined framerate.
                # The following line will automatically add a delay to keep the framerate stable.
                self.clock.tick(self.metadata["render_fps"])
            else:  # rgb_array
                return np.transpose(
                    np.array(pygame.surfarray.pixels3d(self.canvas)), axes=(1, 0, 2)
                )

    def render(self):
        if self.render_mode == "rgb_array":
            return self._render_frame()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()

if __name__ == "__main__":
    render_mode = 'human'
    env = ENV_BASE(render_mode=render_mode)
    observation = env.reset()
    for _ in range(100):
        action = env.action_space.sample()  
        observation, reward, done, _ = env.step(action)
        print(f"[_]: obs [{observation}] | action [{action}] |  reward [{reward}] | done [{done}] ")
        if done:
            print("RESET")
            observation = env.reset()