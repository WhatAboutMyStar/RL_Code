"""
近端策略优化(Proximal Policy Optimization，简称 PPO)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Normal
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import gym
from tqdm import trange
import numpy as np
from collections import deque, namedtuple
import matplotlib.pyplot as plt

env_name = "Pendulum-v0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
batch_size = 32
gamma = 0.9
start_learning = 10

load_model = False #是否加载预训练模型
total_game = 1000
reward_list = []
Transition = namedtuple('Transition', ['state', 'action', 'action_prob', 'reward', 'next_state'])

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.fc = nn.Linear(in_features=env.observation_space.shape[0], out_features=100)
        self.mu = nn.Linear(in_features=100, out_features=env.action_space.shape[0])
        self.sigma = nn.Linear(in_features=100, out_features=env.action_space.shape[0])

    def forward(self, x):
        x = F.relu(self.fc(x))
        mu = 2.0 * torch.tanh(self.mu(x))
        sigma = F.softplus(self.sigma(x))
        return mu, sigma


class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(in_features=env.observation_space.shape[0], out_features=100)
        self.fc2 = nn.Linear(in_features=100, out_features=1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PPO:

    def __init__(self, env, lr, batch_size, gamma):
        self.actor = Actor(env)
        self.critic = Critic(env)
        self.buffer = []

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.batch_size = batch_size
        self.gamma = gamma

    def choose_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            mu, sigma = self.actor(state)

        distribution = Normal(mu, sigma)
        action = distribution.sample()
        action_prob = distribution.log_prob(action)
        action = action.clamp(-2, 2)
        return action.item(), action_prob.item()

    def get_value(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        with torch.no_grad():
            state_value = self.critic(state)
        return state_value.item()

    def save(self):
        torch.save(self.actor.state_dict(), "./actor.pth")
        torch.save(self.critic.state_dict(), "./critic.pth")

    def load(self):
        self.actor.load_state_dict(torch.load("./actor.pth"))
        self.critic.load_state_dict(torch.load("./critic.pth"))

    def store(self, transition):
        self.buffer.append(transition)

    def update(self):

        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.float).view(-1, 1)
        reward = torch.tensor([t.reward for t in self.buffer], dtype=torch.float).view(-1, 1)
        next_state = torch.tensor([t.next_state for t in self.buffer], dtype=torch.float)
        old_action_log_probs = torch.tensor([t.action_prob for t in self.buffer], dtype=torch.float).view(-1, 1)

        reward = (reward - reward.mean()) / (reward.std() + 1e-12)
        with torch.no_grad():
            target_value = reward + self.gamma * self.critic(next_state)

        delta = (target_value - self.critic(state)).detach()

        for i in range(20):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                mu, sigma = self.actor(state[index])
                distribution = Normal(mu, sigma)
                action_log_prob = distribution.log_prob(action[index])
                ratio = torch.exp(action_log_prob - old_action_log_probs[index])

                surr1 = ratio * delta[index]
                surr2 = torch.clamp(ratio, 0.8, 1.2) * delta[index]

                actor_loss = -torch.min(surr1, surr2).mean()
                self.optimizer_actor.zero_grad()
                actor_loss.backward()
                self.optimizer_actor.step()

                critic_loss = F.smooth_l1_loss(self.critic(state[index]), target_value[index])
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        del self.buffer[:]


if __name__ == '__main__':
    env = gym.make(env_name)
    agent = PPO(env, lr, batch_size, gamma)

    if load_model:
        agent.load()

    for i in trange(total_game):
        total_reward = 0
        state = env.reset()
        while True:
            action, action_log_prob = agent.choose_action(state)
            next_state, reward, done, info = env.step([action])
            agent.store(Transition(state, action, action_log_prob, reward, next_state))
            if len(agent.buffer) == 1000:
                agent.update()
            total_reward += reward
            if done:
                reward_list.append(total_reward)
                break
            state = next_state


    agent.save()
    plt.scatter(range(len(reward_list)), reward_list)
    plt.show()

    #test
    total_reward = 0
    state = env.reset()
    agent.load()
    while True:
        env.render()
        action = agent.choose_action(state)
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state

