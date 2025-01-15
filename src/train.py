from gymnasium.wrappers import TimeLimit
from env_hiv import HIVPatient
from memory import PrioritizedReplayBuffer, _gather_per_buffer_attr

env = TimeLimit(
    env=HIVPatient(domain_randomization=False), max_episode_steps=200
)  # The time wrapper limits the number of steps in an episode at 200.
# Now is the floor is yours to implement the agent and train it.
env_rand = TimeLimit(
            env=HIVPatient(domain_randomization=True), max_episode_steps=200
        )


# You have to implement your own agent.
# Don't modify the methods names and signatures, but you can add methods.
# ENJOY!

from typing import *
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random
import pickle

PER = True 
PREV_EPISODES = 0
EPISODES = 1500
MODEL_NAME = "ddqn_512"
if PER:
    SHORT_PATH = "src/" + "torand_logPER_" + MODEL_NAME
else:
    SHORT_PATH = "src/" + "" + MODEL_NAME
LOAD_PATH = SHORT_PATH + f"_best.pth"
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
            nn.LayerNorm(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.LayerNorm(nf),
            nn.Linear(nf, nf),
            nn.LeakyReLU(),
            nn.LayerNorm(nf),
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
        x = torch.log(x)
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
    def __init__(self, gamma=0.99, lr=1e-3, buffer_size=10000, batch_size=128, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.996,
                 grad_clip=1000.0, double_dqn=True, hidden_dim=512,
                 # PER parameters
                per: bool = PER,
                alpha: float = 0.2,
                beta: float = 0.6,
                beta_increment_per_sampling: float = 5e-6,
                prior_eps: float = 1e-6,
):
        # Device: cpu / gpu
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
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

        #self.memory = ReplayBuffer(buffer_size, batch_size)
        # PER memory
        self.per = per
        if per:
            self.prior_eps = prior_eps
            self.alpha = alpha
            self.beta = beta
            self.beta_increment_per_sampling = beta_increment_per_sampling
            self.memory = PrioritizedReplayBuffer(self.state_size, buffer_size, batch_size, alpha, beta, beta_increment_per_sampling)
        else:
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
        # Save model weights
        if self.per:
            _memory = _gather_per_buffer_attr(self.memory)
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "memory": _memory
                    }, path)
            logging.info(f"Agent's state saved to {path}")
            # Save agent config to a JSON file
            config_path = path[:-4] + "_config.pkl"
            agent_config = {
                "device": str(self.device),
                "double_dqn": self.double_dqn,
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
                "grad_clip": self.grad_clip,
                "criterion": self.criterion,
                "rewards": self.rewards,
                "losses": self.losses,
                "per": self.per,
                "alpha": self.alpha,
                "beta": self.beta,
                "beta_increment_per_sampling": self.beta_increment_per_sampling,
                "prior_eps": self.prior_eps
            }
            with open(config_path, "wb") as f:
                pickle.dump(agent_config, f)
            logging.info(f"Agent's config saved to {config_path}")
        else:
            torch.save({
                "q_network": self.q_network.state_dict(),
                "target_network": self.target_network.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                    }, path)
            logging.info(f"Agent's state saved to {path}")
            # Save agent config to a JSON file
            config_path = path[:-4] + "_config.pkl"
            agent_config = {
                "device": str(self.device),
                "double_dqn": self.double_dqn,
                "state_size": self.state_size,
                "action_size": self.action_size,
                "gamma": self.gamma,
                "lr": self.lr,
                "batch_size": self.batch_size,
                "buffer_size" : self.buffer_size,
                "buffer" : list(self.memory.buffer),
                "epsilon": self.epsilon,
                "epsilon_min": self.epsilon_min,
                "epsilon_decay": self.epsilon_decay,
                "hidden_dim": self.hidden_dim,
                "grad_clip": self.grad_clip,
                "criterion": self.criterion,
                "rewards": self.rewards,
                "losses": self.losses,
            }

            with open(config_path, "wb") as f:
                pickle.dump(agent_config, f)
            logging.info(f"Agent's config saved to {config_path}")

    def load(self) -> None:
        checkpoint = torch.load(LOAD_PATH, weights_only=False)
        CONFIG_PATH = LOAD_PATH[:-4] + "_config.pkl"
        with open(CONFIG_PATH, "rb") as f:
            config_checkpoint = pickle.load(f)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_network.eval()
        self.optimizer.load_state_dict(checkpoint["optimizer"])

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.double_dqn = config_checkpoint["double_dqn"]
        self.state_size = config_checkpoint["state_size"]
        self.action_size = config_checkpoint["action_size"]
        self.gamma = config_checkpoint["gamma"]
        self.lr = config_checkpoint["lr"]
        self.batch_size = config_checkpoint["batch_size"]

        self.buffer_size = config_checkpoint["buffer_size"]

        self.epsilon = config_checkpoint["epsilon"]
        self.epsilon_min = config_checkpoint["epsilon_min"]
        self.epsilon_decay = config_checkpoint["epsilon_decay"]

        self.hidden_dim = config_checkpoint["hidden_dim"]

        self.grad_clip = config_checkpoint["grad_clip"]
        self.criterion = config_checkpoint["criterion"]

        self.rewards = config_checkpoint["rewards"]
        self.losses = config_checkpoint["losses"]

        if self.per:
            self.alpha = config_checkpoint["alpha"]
            self.beta = config_checkpoint["beta"]
            self.beta_increment_per_sampling = config_checkpoint["beta_increment_per_sampling"]
            self.prior_eps = config_checkpoint["prior_eps"]
            for key, value in checkpoint['memory'].items():
                if key not in ['sum_tree', 'min_tree']:
                    setattr(self.memory, key, value)
                else:
                    tree = getattr(self.memory, key)
                    setattr(tree, 'capacity', value['capacity'])
                    setattr(tree, 'tree', value['tree'])
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
            self.memory.buffer = deque(config_checkpoint["buffer"], maxlen=self.buffer_size)

        logging.info(f"Agent's state loaded from {LOAD_PATH}")
        logging.info(f"Agent's config loaded from {CONFIG_PATH}")

    def load_specified(self, specified_path):
        checkpoint = torch.load(specified_path, weights_only=False)
        specified_config_path = specified_path[:-4] + "_config.pkl"
        with open(specified_config_path, "rb") as f:
            config_checkpoint = pickle.load(f)

        self.q_network.load_state_dict(checkpoint["q_network"])
        self.target_network.load_state_dict(checkpoint["target_network"])
        self.target_network.eval()
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        logging.info(f"Agent's state loaded from {specified_path}")

        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        self.double_dqn = config_checkpoint["double_dqn"]
        self.state_size = config_checkpoint["state_size"]
        self.action_size = config_checkpoint["action_size"]
        self.gamma = config_checkpoint["gamma"]
        self.lr = config_checkpoint["lr"]
        self.batch_size = config_checkpoint["batch_size"]

        self.buffer_size = config_checkpoint["buffer_size"]

        self.epsilon = config_checkpoint["epsilon"]
        self.epsilon_min = config_checkpoint["epsilon_min"]
        self.epsilon_decay = config_checkpoint["epsilon_decay"]

        self.hidden_dim = config_checkpoint["hidden_dim"]

        self.grad_clip = config_checkpoint["grad_clip"]
        self.criterion = config_checkpoint["criterion"]

        self.rewards = config_checkpoint["rewards"]
        self.losses = config_checkpoint["losses"]
        if self.per:
            self.alpha = config_checkpoint["alpha"]
            self.beta = config_checkpoint["beta"]
            self.beta_increment_per_sampling = config_checkpoint["beta_increment_per_sampling"]
            self.prior_eps = config_checkpoint["prior_eps"]
            for key, value in checkpoint['memory'].items():
                if key not in ['sum_tree', 'min_tree']:
                    setattr(self.memory, key, value)
                else:
                    tree = getattr(self.memory, key)
                    setattr(tree, 'capacity', value['capacity'])
                    setattr(tree, 'tree', value['tree'])
        else:
            self.memory = ReplayBuffer(self.buffer_size, self.batch_size)
            self.memory.buffer = deque(config_checkpoint["buffer"], maxlen=self.buffer_size)
        logging.info(f"Agent's config loaded from {specified_config_path}")

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

    def learn_per(self):
        '''Update the model for PER dqn'''
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch()
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        # PER: importance sampling before average
        elementwise_loss = self._compute_dqn_loss(samples)
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.squeeze().detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray]) -> torch.Tensor:
        '''Return categorical dqn loss.'''
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device)

        curr_q_value = self.q_network(state).gather(1, action)
        if not self.double_dqn:
            next_q_value = self.target_network(next_state).max(dim=1, keepdim=True)[0].detach()
        else:
            next_q_value = self.target_network(next_state).gather(
                1, self.q_network(next_state).argmax(dim=1, keepdim=True)
            ).detach()
        mask = 1 - done
        target = (reward + self.gamma * next_q_value * mask).to(self.device)

        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction='none')

        return elementwise_loss

    def update_target_network(self):
        self.target_network.load_state_dict(self.q_network.state_dict())

from tqdm import tqdm

# Training Loop
def train_dqn(env=env, episodes=1000, update_freq=10, prev_episods=0):
    save_path = SHORT_PATH + f"_{episodes}.pth"
    agent = ProjectAgent()
    if prev_episods > 0:
        prev_path = SHORT_PATH + f"_{prev_episods}.pth"
        agent.load_specified(prev_path)


    for episode in tqdm(range(prev_episods, episodes)):
    #for episode in range(prev_episods, episodes):
        state, _ = env.reset()
        total_reward = 0
        losses = []
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.act(state, use_random=True)
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

        checkpoint_freq = 100
        if (episode % checkpoint_freq == 0) and (episode > prev_episods):
            agent.save(SHORT_PATH + f"_{episode}.pth")

        print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, Epsilon: {agent.epsilon:.3f}, loss: {agent.losses[-1]}")

    env.close()
    agent.save(save_path)

def train_dqn_per(env=env, episodes=1000, update_freq=10, prev_episods=0):

    save_path = SHORT_PATH + f"_{episodes}.pth"
    agent = ProjectAgent()
    if prev_episods > 0:
        prev_path = SHORT_PATH + f"_{prev_episods}.pth"
        agent.load_specified(prev_path)

    best_reward = 400

    last_5_evals = []
    last_5_evals_rand = []

    for episode in tqdm(range(prev_episods, episodes)):
    #for episode in range(prev_episods, episodes):    
        if episode > 450:
            agent.lr = 3e-4
            state, _ = env_rand.reset()
        else:
            state, _ = env.reset()
        n_etapes = 0
        total_reward = 0
        losses = []
        done = False
        truncated = False

        while not done and not truncated:
            action = agent.select_action(state)
            if episode > 450:
                next_state, reward, done, truncated, _  = env_rand.step(action)
            else:
                next_state, reward, done, truncated, _  = env.step(action)
            reward /= reward_scaler
            transition = [state, action, reward, next_state, done]
            agent.memory.store(*transition)

            state = next_state
            total_reward += reward
            n_etapes += 1
            # If training is available:
            if len(agent.memory) >= agent.batch_size:
                loss = agent.learn_per()
                losses.append(loss)

        agent.append_rewards(total_reward)
        if losses:
            agent.append_losses(np.mean(losses))
        else:
            agent.append_losses(-1.)
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        if episode % update_freq == 0:
            agent.update_target_network()

        checkpoint_freq = 50
        if (episode % checkpoint_freq == 0) and (episode > prev_episods):
            agent.save(SHORT_PATH + f"_{episode}.pth")
        
        if (episode > 300) and (episode % 10 == 0):
            #evaluate one episode
            obs, info = env.reset()
            done = False
            truncated = False
            eval_reward = 0
            while not done and not truncated:
                action = agent.act(obs)
                obs, reward, done, truncated, _ = env.step(action)
                eval_reward += reward
            last_5_evals.append(eval_reward)

            #evaluate one episode
            obs, info = env_rand.reset()
            done = False
            truncated = False
            eval_reward_rand = 0
            while not done and not truncated:
                action = agent.act(obs)
                obs, reward, done, truncated, _ = env_rand.step(action)
                eval_reward_rand += reward
            last_5_evals_rand.append(eval_reward_rand)


            
            if len(last_5_evals) > 5:
                last_5_evals.pop(0)
                last_5_evals_rand.pop(0)
                mean_5_evals = np.mean(last_5_evals)
                mean_5_evals_rand = np.mean(last_5_evals_rand)
                if (mean_5_evals > 200) and (mean_5_evals_rand > 200) and (mean_5_evals + mean_5_evals_rand > best_reward):
                    best_reward = mean_5_evals + mean_5_evals_rand
                    agent.save(SHORT_PATH + f"_best.pth")


                print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, loss: {agent.losses[-1]}, eval:{mean_5_evals/1e8}, eval_rand:{mean_5_evals_rand/1e8}")
            else:
                print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, loss: {agent.losses[-1]}")
        else:
            print(f"Episode {episode + 1}/{episodes}, Reward: {total_reward}, loss: {agent.losses[-1]}")

    env.close()
    agent.save(save_path)

def train_(env=env, episodes=1000, update_freq=10, prev_episods=0):
    if PER:
        return train_dqn_per(env=env, episodes=episodes, update_freq=update_freq, prev_episods=prev_episods)
    else:
        return train_dqn(env=env, episodes=episodes, update_freq=update_freq, prev_episods=prev_episods)
    
if __name__ == "__main__":
    train_(env=env, episodes=EPISODES, update_freq=5, prev_episods=PREV_EPISODES)