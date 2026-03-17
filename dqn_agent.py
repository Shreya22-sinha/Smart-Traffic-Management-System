import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Neural Network for Q-learning
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(2, 24),
            nn.ReLU(),
            nn.Linear(24, 24),
            nn.ReLU(),
            nn.Linear(24, 3)  # 3 possible actions
        )

    def forward(self, x):
        return self.fc(x)

class DQNAgent:
    def __init__(self):
        self.model = DQN()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.memory = []
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_decay = 0.995

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 2)
        state = torch.FloatTensor(state)
        q_values = self.model(state)
        return torch.argmax(q_values).item()

    def store(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def train(self):
        if len(self.memory) < 10:
            return

        batch = random.sample(self.memory, 10)

        for state, action, reward, next_state in batch:
            state = torch.FloatTensor(state)
            next_state = torch.FloatTensor(next_state)

            target = reward + self.gamma * torch.max(self.model(next_state))
            output = self.model(state)[action]

            loss = (target - output) ** 2

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        self.epsilon *= self.epsilon_decay
