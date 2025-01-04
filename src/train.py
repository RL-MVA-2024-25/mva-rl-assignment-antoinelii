from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random

# Define the Q-Network
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Replay Buffer for Experience Replay
class ReplayBuffer:
    def __init__(self, buffer_size, batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self):
        batch = random.sample(self.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.tensor(np.array(states), dtype=torch.float32),
            torch.tensor(np.array(actions), dtype=torch.long),
            torch.tensor(np.array(rewards), dtype=torch.float32),
            torch.tensor(np.array(next_states), dtype=torch.float32),
            torch.tensor(np.array(dones), dtype=torch.float32),
        )

    def size(self):
        return len(self.buffer)

# DQN Agent
class ProjectAgent:
    def __init__(self, gamma=0.99, lr=1e-3, buffer_size=1000, batch_size=64, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995):
        self.state_size = env.unwrapped.observation_space.shape[0]
        self.action_size = env.unwrapped.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.q_network = QNetwork(self.state_size, self.action_size)
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.memory = ReplayBuffer(buffer_size, batch_size)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state)
        return torch.argmax(q_values).item()
    
    def save(self, path):
        with open(path, "wb") as f:
            torch.save(self, f)
        # Save model weights and configuration
        torch.save({
            "state_size": self.state_size,
            "action_size": self.action_size,
            "gamma": self.gamma,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "buffer_size" : self.buffer_size,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "q_network": self.q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "criterion": self.criterion.state_dict(),
                }, path)
        print(f"Agent's state saved to {path}")

    def load(self):
        checkpoint = torch.load("dqn_100.pth")
        self.state_size = checkpoint["state_size"]
        self.action_size = checkpoint["action_size"]
        self.gamma = checkpoint["gamma"]
        self.lr = checkpoint["lr"]
        self.batch_size = checkpoint["batch_size"]

        self.epsilon = checkpoint["epsilon"]
        self.epsilon_min = checkpoint["epsilon_min"]
        self.epsilon_decay = checkpoint["epsilon_decay"]

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network = QNetwork(self.state_size, self.action_size)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.criterion.load_state_dict(checkpoint["criterion"])
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

    def learn(self):
        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

#from tqdm import tqdm

# Training Loop
def train_dqn(env=env, episodes=100, update_freq=10):
    save_path = f"dqn_{episodes}.pth"

    agent = ProjectAgent()
    rewards = []

    for episode in range(episodes):
        state, _ = env.reset()
        total_reward = 0
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.act(state)
            next_state, reward, done, truncated, _  = env.step(action)
            agent.memory.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            agent.learn()

        rewards.append(total_reward)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if episode % update_freq == 0:
            agent.update_target_network()

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}")

    env.close()
    agent.save(save_path)
    return rewards, agent

if __name__ == "__main__":
    rewards, trained_agent = train_dqn(env=env, episodes=100, update_freq=10)