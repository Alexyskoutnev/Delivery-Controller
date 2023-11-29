import torch
import torch.nn as nn
import torch.optim as optim


from RL_Pizza_Delivery.algo.MCTS import MCTS
from RL_Pizza_Delivery.env.env_obstacles import ENV_OBSTACLE


class ValueNetwork(nn.Module):
    def __init__(self, input_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_value_network(value_network, states, targets, num_epochs=100, learning_rate=0.001):
    optimizer = optim.Adam(value_network.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        predicted_values = value_network(states)
        loss = criterion(predicted_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss.item()}")

# Main training loop

num_episodes = 100
num_mcts_iterations = 1000
config = {"epochs": 10000, "lr": 1e-3, 'potholes': 0, 'traffic_jams': 0
            , 'map_size': (5, 5), 'render_mode': None, 'epoch_print_cnt': 500}
env = ENV_OBSTACLE(map_size=config['map_size'], render_mode=config['render_mode'], potholes=config['potholes'], traffic_jams=config['traffic_jams'])
 
mcts = MCTS(env, input_size=env.observation_dim)
value_network = ValueNetwork(input_size=env.observation_dim)


for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        action = mcts.search(env, state, num_iterations=num_mcts_iterations)
        next_state, _, done, _ = env.step(action)

        # Collect data for training the value network
        state_tensor = torch.tensor(state, dtype=torch.float32).view(1, -1)
        target_value = mcts.value_network(state_tensor).item()
        target_tensor = torch.tensor([[target_value]], dtype=torch.float32)

        train_value_network(value_network, state_tensor, target_tensor)

        state = next_state

    print(f"Episode {episode + 1} finished.")
