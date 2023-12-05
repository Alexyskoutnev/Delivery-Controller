import numpy as np
import torch
import torch.optim as optim
from tensorboardX import SummaryWriter

from RL_Pizza_Delivery.env.env_obstacles_mcts import ENV_OBSTACLE
from RL_Pizza_Delivery.utils.buffer import ExperienceBuffer
from RL_Pizza_Delivery.algo.MCTS.mcts import MCTSNode, select_child, expand, simulate, backpropagate

# Define your training parameters
iterations_per_episode = 1000
num_episodes = 500

# Environment parameters
potholes, traffic_jams = 1, 1
map_size = (5, 5)
render_mode = 'human'
max_moves = map_size[0] * 10  # Adjust based on your environment

# Create environment
env = ENV_OBSTACLE(map_size=map_size, render_mode=render_mode, potholes=potholes, traffic_jams=traffic_jams)

# Initialize MCTS parameters
exploration_weight = 1.0

# Training loop
for episode in range(1, num_episodes + 1):
    total_reward = 0.0
    node = MCTSNode(state=env.reset())

    for _ in range(iterations_per_episode):
        current_node = node
        while not env.is_terminal() and not current_node.untried_actions and current_node.children:
            current_node = select_child(current_node)

        if current_node.untried_actions:
            action = np.random.choice(current_node.untried_actions)
            new_node, reward, done = expand(env, current_node, action)  # Pass env as an argument
        else:
            done = env.is_terminal()
            reward = 0.0
        reward += simulate(env, new_node, max_moves)  # Pass env as an argument
        backpropagate(new_node, reward)

        # Choose action based on MCTS results
        best_action = max(node.children, key=lambda child: child.value / (child.visits + 1e-6))
        action = best_action.action  # You need to modify this based on your implementation

        # Take action in the environment
        observation, reward, done, _ = env.step(action)
        total_reward += reward

        if done:
            break

    # Log total reward for the episode
    print(f"Episode {episode}, Total Reward: {total_reward}")

# Close the environment
env.close()
