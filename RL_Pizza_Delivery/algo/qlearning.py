import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from RL_Pizza_Delivery.utils.buffer import Experience

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
    def __init__(self, env, buffer, lr=0.001, gamma=0.99, epsilon=0.1) -> None:
        self.env = env
        self.input_size = env.observation_dim
        self.output_size = env.action_dim
        self.net = QNetwork(self.input_size, self.output_size)
        self.net_target = QNetwork(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.espilon = epsilon
        self.eval = False
        self.buffer = buffer
        self._reset()

    @torch.no_grad()
    def play_step(self, epsilon=0.0, device="cpu"):
        done_reward = None

        if np.random.random() < epsilon:
            action = self.env.action_space.sample()
        else:
            state = torch.tensor(self.state, dtype=torch.float32).to(device)
            action = self.select_action(state)

        # do step in the environment
        new_state, reward, is_done, _ = self.env.step(action)
        self.total_reward += reward

        exp = Experience(self.state, action, reward,
                         is_done, new_state)
        self.buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self._reset()
        return done_reward

    def _reset(self):
        self.state = self.env.reset()
        self.total_reward = 0.0

    def select_action(self, state, espilon=0.1):
        if self.eval:
            with torch.no_grad():
                return torch.argmax(self.net(state)).item()
        if np.random.rand() < espilon:
            return np.random.randint(0, self.output_size)
        else:
            with torch.no_grad():
                return torch.argmax(self.net(state)).item()

    def update(self, batch):
        state, action, reward, done, next_state = batch
        self.optimizer.zero_grad()
        #========Conver to Tensor=======
        states = torch.tensor(state)
        next_states = torch.tensor(next_state)
        actions = torch.tensor(action)
        rewards = torch.tensor(reward)
        dones =  torch.BoolTensor(done)
        #===============================
        # q_values = self.net(states)
        state_action_values = self.net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            next_state_values = self.net_target(next_states).max(1)[0]
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()
        target = rewards + self.gamma * next_state_values
        loss = self.criterion(state_action_values, target)
        loss.backward()
        self.optimizer.step()
        return loss