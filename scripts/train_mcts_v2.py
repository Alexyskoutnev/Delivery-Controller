import numpy as np
from RL_Pizza_Delivery.env.env_obstacles_mcts import ENV_OBSTACLE
from RL_Pizza_Delivery.algo.MCTS.mcts import MCTSNode, select_child, expand, simulate, backpropagate

# Define your training parameters
iterations_per_episode = 1000
num_episodes = 500

# Environment parameters
potholes = 5
traffic_jams = 0
map_size = (5, 5)
render_mode = 'human'
max_moves = map_size[0] // 2  # Adjust based on your environment

# Create environment
env = ENV_OBSTACLE(map_size=map_size, render_mode=render_mode, potholes=potholes, traffic_jams=traffic_jams)

# Initialize MCTS parameters
exploration_weight = 1.0

# Training loop
for episode in range(1, num_episodes + 1):
    total_reward = 0.0
    root = MCTSNode(state=env.reset())

    for _ in range(iterations_per_episode):
        node = root
        # Selection
        while not env.is_terminal() and not node.untried_actions and node.children:
            node = select_child(node)

        # Expansion
        if node.untried_actions:
            action = np.random.choice(node.untried_actions)
            new_node, reward, done = expand(env, node, action)
        else:
            done = env.is_terminal()
            reward = 0.0

        # Simulation
        reward += simulate(env, node, max_moves)

        # Backpropagation
        backpropagate(new_node, reward)

    # Choose action based on MCTS results
    best_action = max(root.children, key=lambda child: child.value / (child.visits + 1e-6)).action
    # Take action in the environment
    observation, reward, done, _ = env.step(best_action)
    total_reward += reward

    # Log total reward for the episode
    print(f"Episode {episode}, Total Reward: {total_reward}")

# Close the environment
env.close()
