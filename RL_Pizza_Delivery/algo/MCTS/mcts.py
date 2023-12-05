import numpy as np

class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action  # Add this line to include the action attribute
        self.visits = 0
        self.value = 0.0
        self.untried_actions = list(range(4))  # Assuming 4 possible actions (UP, DOWN, LEFT, RIGHT)

def select_child(node):
    exploration_weight = 1.0  # You may adjust this parameter
    child_scores = [
        child.value / (child.visits + 1e-6) + exploration_weight * np.sqrt(np.log(node.visits) / (child.visits + 1e-6))
        for child in node.children
    ]
    return node.children[np.argmax(child_scores)]

def expand(env, node, action):
    """
    Expand the MCTS tree by adding a new node based on the given action.

    Args:
        env (gym.Env): The environment.
        node (MCTSNode): The current node in the MCTS tree.
        action (int): The action to take.

    Returns:
        MCTSNode: The new node created after taking the given action.
        float: The reward obtained from the environment after taking the action.
        bool: Whether the episode is finished after taking the action.
    """
    new_state, reward, done, _ = env.step(action)
    new_node = MCTSNode(state=new_state, parent=node, action=action)
    node.children.append(new_node)
    return new_node, reward, done

def simulate(env, node, max_moves):
    """
    Simulate a rollout from the given node until a terminal state or a maximum number of moves.

    Args:
        env (gym.Env): The environment.
        node (MCTSNode): The starting node for the simulation.
        max_moves (int): The maximum number of moves for the simulation.

    Returns:
        float: The cumulative reward obtained during the simulation.
    """
    total_reward = 0.0
    current_state = node.state
    for _ in range(max_moves):
        action = np.random.choice(env.action_space.n)
        new_state, reward, done, _ = env.step(action)
        total_reward += reward
        if done:
            break
    return total_reward

def backpropagate(node, reward):
    while node is not None:
        node.visits += 1
        node.value += reward
        node = node.parent

def mcts(env, iterations=1000):
    root = MCTSNode(state=env.reset())

    for _ in range(iterations):
        node = root
        # Selection
        while not env.is_terminal() and not node.untried_actions and node.children:
            node = select_child(node)

        # Expansion
        if node.untried_actions:
            action = np.random.choice(node.untried_actions)
            node, reward, done = expand(node, action)
        else:
            done = env.is_terminal()
            reward = 0.0

        # Simulation
        reward += simulate(node)

        # Backpropagation
        backpropagate(node, reward)

    # Return the action with the highest average value
    best_action = max(root.children, key=lambda child: child.value / (child.visits + 1e-6))
    return best_action
