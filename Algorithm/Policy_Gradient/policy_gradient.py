import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from torch.distributions import Categorical
from tqdm import trange
from collections import deque
import numpy as np
import matplotlib.pyplot as plt

buffer = deque(maxlen=200*100)

class Net(nn.Module):
    def __init__(self, env):
        super(Net, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.fc1 = nn.Linear(in_features=self.state_space, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=self.action_space)

    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x



if __name__ == '__main__':
    env = gym.make("CartPole-v0")
    # env = gym.make("MountainCar-v0")

    policy = Net(env)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    gamma = 0.98
    batch_size = 128

    loss_curve = []
    reward_list = []
    for i in trange(300):
        all_reward = 0
        state = env.reset()
        saved_log_probs = []
        rewards = []

        while True:
            state = torch.from_numpy(state).float().unsqueeze(0)

            #根据概率选择一个action
            probs = policy(state)
            m = Categorical(probs)
            action = m.sample()
            saved_log_probs.append(m.log_prob(action))

            #与环境交互
            next_state, reward, done, info = env.step(action.item())
            rewards.append(reward)

            all_reward += reward

            if done:
                reward_list.append(all_reward)
                break
            state = next_state

        policy_loss = []
        returns = []
        R = 0
        for r in rewards[::-1]:
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.Tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-12)

        for log_prob, r in zip(saved_log_probs, returns):
            policy_loss.append(-log_prob * r)
        optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        optimizer.step()
        loss_curve.append(policy_loss)

    torch.save(policy.state_dict(), "./pg_model.pth")
    plt.scatter(range(len(reward_list)), reward_list)
    plt.show()


    plt.plot(loss_curve)
    plt.show()


    #test
    testAgent = Net(env)
    testAgent.load_state_dict(torch.load("./pg_model.pth"), strict=False)
    state = env.reset()
    while True:
        env.render()
        action = int(testAgent(torch.Tensor(state.reshape(1, -1))).argmax(1)[0])
        next_state, reward, done, info = env.step(action)
        if done:
            break
        state = next_state