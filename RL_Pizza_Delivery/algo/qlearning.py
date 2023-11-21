import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class QNetwork(nn.Module):
    def __init__(self, input_size, output_size) -> None:
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class QAgent(object):
    def __init__(self, input_size, output_size, lr=0.001, gamma=0.95, epsilon=0.1) -> None:
        self.q_network = QNetwork(input_size, output_size)
        self.output_size = output_size
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.espilon =epsilon

    def select_action(self, state):
        if np.random.rand() < self.espilon:
            return np.random.randint(0, self.output_size)
        else:
            with torch.no_grad():
                return torch.argmax(self.q_network(state)).item()

    def update(self, state, action, reward, next_state, done):
        self.optimizer.zero_grad()
        q_values = self.q_network(state)
        next_q_values = self.q_network(next_state).max().detach()
        target = reward + (1 - done) * self.gamma * next_q_values
        loss = self.criterion(q_values, target)
        loss.backward()
        self.optimizer.step()