import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from tqdm import trange
import numpy as np
from collections import deque

env_name = "Pendulum-v0"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
lr = 0.001
batch_size = 32
gamma = 0.9
start_learning = 10

load_model = True
total_game = 500
buffer = deque(maxlen=10000)

class Actor(nn.Module):
    def __init__(self, env):
        super(Actor, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]
        self.fc1 = nn.Linear(in_features=self.state_space, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=self.action_space)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x)) * 2
        return x


class Critic(nn.Module):
    def __init__(self, env):
        super(Critic, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.shape[0]

        self.fc1 = nn.Linear(in_features=self.state_space + self.action_space, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DDPG:
    def __init__(self, env, batch_size, gamma, lr):
        self.actor = Actor(env).to(device)
        self.critic = Critic(env).to(device)

        self.actor_target = Actor(env).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic_target = Critic(env).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.buffer = deque(maxlen=10000)
        self.batch_size = batch_size
        self.gamma = gamma

    def choose_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(device)
        action = self.actor(state)
        return action.cpu().detach().numpy().flatten()

    def save(self):
        torch.save(self.actor.state_dict(), "./actor.pth")
        torch.save(self.critic.state_dict(), "./critic.pth")

    def load(self):
        self.actor.load_state_dict(torch.load("./actor.pth"))
        self.critic.load_state_dict(torch.load("./critic.pth"))

    def sample(self):
        sample_index = np.random.choice(len(self.buffer), batch_size, replace=False)

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_done = []

        for index in sample_index:
            sample = self.buffer[index]

            batch_state.append(sample[0])
            batch_action.append(sample[1])
            batch_reward.append(sample[2])
            batch_next_state.append(sample[3])
            batch_done.append(sample[4])

        batch_state = torch.FloatTensor(batch_state).to(device)
        batch_action = torch.FloatTensor(batch_action).to(device)
        batch_reward = torch.FloatTensor(batch_reward).to(device).view(-1, 1)
        batch_next_state = torch.FloatTensor(batch_next_state).to(device)
        batch_done = torch.FloatTensor(1 - np.array(batch_done) * 1).to(device).view(-1, 1)

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def update(self):
        for i in range(20):
            batch_state, batch_action, batch_reward, batch_next_state, batch_done = self.sample()

            target_Q = self.critic_target(batch_next_state, self.actor_target(batch_next_state))
            target_Q = batch_reward + (batch_done * gamma * target_Q)

            current_Q = self.critic(batch_state, batch_action)

            critic_loss = F.mse_loss(current_Q, target_Q)

            self.optimizer_critic.zero_grad()
            critic_loss.backward()
            self.optimizer_critic.step()

            actor_loss = -self.critic(batch_state, self.actor(batch_state)).mean()

            self.optimizer_actor.zero_grad()
            actor_loss.backward()
            self.optimizer_actor.step()

            if i % 5 == 0:
                self.actor_target.load_state_dict(self.actor.state_dict())
                self.critic_target.load_state_dict(self.critic.state_dict())

if __name__ == '__main__':
    env = gym.make(env_name)
    total_reward = 0
    state = env.reset()
    agent = DDPG(env, batch_size, gamma, lr)
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

