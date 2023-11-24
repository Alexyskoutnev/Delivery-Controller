import numpy as np
import torch
from RL_Pizza_Delivery.algo.qlearning import QAgent

class ENV(object):
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.state_size = grid_size * grid_size
        self.action_size = 4
        self.action_dim = 4
        self.grid = np.zeros((grid_size, grid_size))
        self.start = np.array([0, 0], dtype=np.float32)
        self.observation_dim = 2
        self.state = self.start
        self.goal = (grid_size -1, grid_size - 1)
        self.obstacles = [(1, 1), (2, 2), (3, 1)]
    
    def reset(self):
        self.grid = np.zeros((self.grid_size, self.grid_size))
        for obstacle in self.obstacles:
            self.grid[obstacle[0]][obstacle[1]] = -1
        return np.array(self.start, dtype=np.float32)

    def step(self, action):
        next_state = self.state

        if action == 0 and self.state[0] > 0:
            next_state[0] -= 1  # Move up
        elif action == 1 and self.state[0] < self.grid_size - 1:
            next_state[0] += 1  # Move down
        elif action == 2 and self.state[1] > 0:
            next_state[1] -= 1  # Move left
        elif action == 3 and self.state[1] < self.grid_size - 1:
            next_state[1] += 1  # Move right

        reward = -1  # Move cost
        if tuple(next_state) in self.obstacles:
            reward = -5  # Penalty for hitting an obstacle
            next_state = self.state

        done = False
        if tuple(next_state) == self.goal:
            reward = 10  # Goal reached
            done = True
        
        self.state = next_state

        return next_state, reward, done, {}

if __name__ == "__main__":
    # Training parameters
    grid_size = 10
    state_size = grid_size * grid_size
    action_size = 4
    learning_rate = 0.001
    gamma = 0.99
    epsilon_decay = 0.995
    epsilon_min = 0.01
    episodes = 1

    env = ENV(grid_size)
    input_size, output_size = env.observation_dim, env.action_dim
    agent = QAgent(input_size, output_size)

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        epsilon = max(epsilon_min, epsilon_decay**episode)
        while True:
            action = agent.select_action(torch.tensor(state, dtype=torch.float32))
            next_state, reward, done, _ = env.step(action)
            agent.update(torch.tensor(state, dtype=torch.float32), action, reward, torch.tensor(next_state, dtype=torch.float32), done)
            state = next_state
            total_reward += reward
            print(f"state : [{state}], action : [{action}], next state: [{next_state}], reward: [{reward}]")
            if done:
                break
        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")
    
    # Test the trained agent
    state = env.reset()
    path = [state]

    while True:
        action = agent.select_action(torch.tensor(state, dtype=torch.float32))
        next_state, _, done, _ = env.step(action)
        path.append(next_state)
        state = next_state
        breakpoint()
        if done:
            break

    print("Optimal Path:", path)