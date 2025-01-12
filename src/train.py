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
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

PREV_EPISODES = 0
EPISODES = 500
SHORT_PATH = "src/ddqn_512"
LOAD_PATH = SHORT_PATH + f"_{EPISODES}.pth"
reward_scaler = 1e8

import logging

# Configure the logging
logging.basicConfig(
    level=logging.INFO,  # Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Define the Q-Network
class QNetwork(nn.Module):
    #def __init__(self, state_size, action_size):
    #    super(QNetwork, self).__init__()
    #    self.fc1 = nn.Linear(state_size, 512)
    #    self.fc2 = nn.Linear(512, 512)
    #    self.fc3 = nn.Linear(512, action_size)

    #def forward(self, x):
    #    x = torch.relu(self.fc1(x))
    #    x = torch.relu(self.fc2(x))
    #    return self.fc3(x)
    
    def __init__(self, in_dim: int, nf: int, out_dim: int):
        """Initialization."""
        super(QNetwork, self).__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_dim, nf), 
            nn.LeakyReLU(),
            nn.Linear(nf, nf), 
            nn.LeakyReLU(),
            nn.Linear(nf, nf), 
            nn.LeakyReLU(),
            nn.Linear(nf, out_dim)
        )
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight.data)
                nn.init.zeros_(m.bias.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        return self.layers(x)

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
    def __init__(self, gamma=0.99, lr=2e-4, buffer_size=10000, batch_size=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 grad_clip=1000.0, double_dqn=True, hidden_dim=512,
):
        # Device: cpu / gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info(f'device: {self.device}')
        self.double_dqn = double_dqn

        self.state_size = env.unwrapped.observation_space.shape[0]
        self.action_size = env.unwrapped.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.hidden_dim = hidden_dim
        self.q_network = QNetwork(self.state_size, self.hidden_dim, self.action_size).to(self.device)
        self.target_network = QNetwork(self.state_size, self.hidden_dim, self.action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer = optim.Adam(self.q_network.parameters(), lr=lr)
        self.grad_clip = grad_clip
        self.criterion = "smooth_l1_loss"
        self.memory = ReplayBuffer(buffer_size, batch_size)

        self.rewards = []
        self.losses = []

    def act(self, observation: np.ndarray, use_random: bool=False) -> int:
        if (use_random) & (random.random() < self.epsilon):
            return random.randint(0, self.action_size - 1)
        observation = torch.tensor(observation, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            selected_action = self.q_network(observation).argmax()

        return selected_action.detach().cpu().numpy()
    
    def save(self, path: str) -> None:
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
            "hidden_dim": self.hidden_dim,
            "q_network": self.q_network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "grad_clip": self.grad_clip,
            "criterion": self.criterion,
            "rewards": self.rewards,
            "losses": self.losses,
                }, path)
        print(f"Agent's state saved to {path}")

    def load(self) -> None:
        checkpoint = torch.load(LOAD_PATH)
        self.state_size = checkpoint["state_size"]
        self.action_size = checkpoint["action_size"]
        self.gamma = checkpoint["gamma"]
        self.lr = checkpoint["lr"]
        self.batch_size = checkpoint["batch_size"]

        self.epsilon = checkpoint["epsilon"]
        self.epsilon_min = checkpoint["epsilon_min"]
        self.epsilon_decay = checkpoint["epsilon_decay"]

        self.hidden_dim = checkpoint["hidden_dim"]
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.grad_clip = checkpoint["grad_clip"]
        self.criterion = checkpoint["criterion"]
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.rewards = checkpoint["rewards"]
        self.losses = checkpoint["losses"]

    def load_specified(self, specified_path):
        checkpoint = torch.load(specified_path)
        self.state_size = checkpoint["state_size"]
        self.action_size = checkpoint["action_size"]
        self.gamma = checkpoint["gamma"]
        self.lr = checkpoint["lr"]
        self.batch_size = checkpoint["batch_size"]

        self.epsilon = checkpoint["epsilon"]
        self.epsilon_min = checkpoint["epsilon_min"]
        self.epsilon_decay = checkpoint["epsilon_decay"]

        self.hidden_dim = checkpoint["hidden_dim"]
        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.grad_clip = checkpoint["grad_clip"]
        self.criterion = checkpoint["criterion"]
        self.memory = ReplayBuffer(self.buffer_size, self.batch_size)

        self.rewards = checkpoint["rewards"]
        self.losses = checkpoint["losses"]
    
    def append_rewards(self, reward):
        self.rewards.append(reward)

    def append_losses(self, loss):
        self.losses.append(loss)

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            selected_action = self.q_network(state).argmax()

        return selected_action.detach().cpu().numpy()
    
    def learn(self):
        if self.memory.size() < self.batch_size:
            return

        states, actions, rewards, next_states, dones = self.memory.sample()
        device = self.device  # for shortening the following lines
        states = torch.FloatTensor(states).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        actions = torch.LongTensor(actions.reshape(-1, 1)).to(device)
        rewards = torch.FloatTensor(rewards.reshape(-1, 1)).to(device)
        dones = torch.FloatTensor(dones.reshape(-1, 1)).to(device)

        curr_q_values = self.q_network(states).gather(1, actions)
        if not self.double_dqn:
            next_q_values = self.target_network(next_states).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_values = self.target_network(next_states).gather(
                1, self.q_network(next_states).argmax(dim=1, keepdim=True)
            ).detach()
        mask = 1 - dones
        targets = (rewards + self.gamma * next_q_values * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_values, targets, reduction='none')
        loss = torch.mean(elementwise_loss)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        return loss.detach().cpu().numpy()

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

#from tqdm import tqdm

# Training Loop
def train_dqn(env=env, episodes=1000, update_freq=10, prev_episods=0):
    save_path = SHORT_PATH + f"_{episodes}.pth"
    agent = ProjectAgent()
    if prev_episods > 0:
        prev_path = SHORT_PATH + f"_{prev_episods}.pth"
        agent.load_specified(prev_path)


    #for episode in tqdm(range(episodes-prev_episods)):
    for episode in range(episodes-prev_episods):
        state, _ = env.reset()
        total_reward = 0
        losses = []
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.select_action(state)
            next_state, reward, done, truncated, _  = env.step(action)
            reward /= reward_scaler
            agent.memory.add(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward

            loss = agent.learn()
            if loss is not None:
                losses.append(loss)

        agent.append_rewards(total_reward)
        agent.append_losses(np.mean(losses))
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if episode % update_freq == 0:
            agent.update_target_network()
        
        log_freq = 10
        if episode % log_freq == 0:
            pass

        logging.info(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, loss: {agent.losses[-1]}")

    env.close()
    agent.save(save_path)
    return agent.rewards, agent

if __name__ == "__main__":
    rewards, trained_agent = train_dqn(env=env, episodes=EPISODES, update_freq=5, prev_episods=PREV_EPISODES)