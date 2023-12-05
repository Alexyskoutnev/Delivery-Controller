# from RL_Pizza_Delivery.env.env import ENV_BASE
import math
import numpy as np
import torch
import torch.nn as nn

class Node:
    def __init__(self, state, parent=None):
        self.state = state
        self.action = None
        self.parent = parent
        self.children = []
        self.visits = 0
        self.value = 0


class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class MCTS:
    def __init__(self, env, exploration_weight=1.0, input_size=2):
        self.exploration_weight = exploration_weight
        self.value_network = ValueNetwork(input_size)
        self.env = env
        self.max_steps = 100

    def select(self, node):
        while node.children:
            node = max(node.children, key=lambda child: child.value + self.exploration_weight * math.sqrt(math.log(node.visits) / (child.visits + 1)))
        return node

    def expand(self, node):
        actions = range(self.env.action_space.n)
        for action in actions:
            next_state, _, _, _ = self.env.step(action)
            child_node = Node(next_state, parent=node)
            node.children.append(child_node)
        return node.children[np.random.choice(len(node.children))]

    def simulate(self, node):
        total_reward = 0
        for _ in range(self.max_steps):
            action = np.random.choice(self.env.action_space.n)
            _, reward, done, _ = self.env.step(action)
            total_reward += reward
            if done:
                break
        return total_reward

    def backpropagate(self, node, reward):
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent

    def search(self, env, root_state, num_iterations):
        root = Node(root_state)

        for _ in range(num_iterations):
            node = self.select(root)
            if not node.children and node.visits > 0:
                node = self.expand(node)

            state_tensor = torch.tensor(node.state, dtype=torch.float32).view(1, -1)
            value = self.value_network(state_tensor).item()
            reward = self.simulate(node) + value  # Combine reward with value estimate
            self.backpropagate(node, reward)

        best_child = max(root.children, key=lambda child: child.visits)
        return best_child.state
