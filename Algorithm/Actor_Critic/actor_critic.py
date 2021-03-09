import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from torch.distributions import Categorical

#超参数
gamma = 0.99
lr = 0.01
env_name = "CartPole-v0"
total_game = 1000 #开启游戏玩的局数

buffer = []

#评估数据
reward_list = []
loss_curve = []

class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n

        #Actor
        self.fc1 = nn.Linear(in_features=self.state_space, out_features=128)
        self.actor_pi = nn.Linear(in_features=128, out_features=self.action_space)

        #Critic
        self.critic_value = nn.Linear(in_features=128, out_features=1)

    def actor(self, state):
        x = F.relu(self.fc1(state))
        x = self.actor_pi(x)
        x = F.softmax(x, dim=1)
        return x

    def critic(self, state):
        share = self.fc1(state)
        x = self.critic_value(share)
        return x


if __name__ == '__main__':
    env = gym.make(env_name)
    actor_critic = ActorCritic(env)
    optimizer = torch.optim.Adam(actor_critic.parameters(), lr=lr)


    for i in trange(total_game):
        total_reward = 0
        state = env.reset()

        while True:
            probs = actor_critic.actor(torch.from_numpy(state).float().unsqueeze(0))
            m = Categorical(probs)
            action = m.sample().item()
            next_state, reward, done, info = env.step(action)

            buffer.append([state, action, reward, next_state, done])
            total_reward += reward

            if done:
                reward_list.append(total_reward)
                break

            state = next_state

        batch_state = []
        batch_action = []
        batch_reward = []
        batch_next_state = []
        batch_done = []

        for batch in buffer:
            s, a, r, ns, d = batch
            batch_state.append(s)
            batch_action.append([a])
            batch_reward.append([r])
            batch_next_state.append(ns)
            batch_done.append([0 if d else 1])

        batch_state = torch.tensor(batch_state, dtype=torch.float)
        batch_action = torch.tensor(batch_action)
        batch_reward = torch.tensor(batch_reward)
        batch_next_state = torch.tensor(batch_next_state, dtype=torch.float)
        batch_done = torch.tensor(batch_done, dtype=torch.float)

        buffer = []

        td_target = batch_reward + gamma * actor_critic.critic(batch_next_state) * batch_done

        delta = td_target - actor_critic.critic(batch_state)
        actor_action = actor_critic.actor(batch_state).gather(1, batch_action)
        loss = (-torch.log(actor_action) * delta + F.smooth_l1_loss(actor_critic.critic(batch_state), td_target)).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_curve.append(loss)

    torch.save(actor_critic.state_dict(), "./actor_critic_model.pth")
    plt.scatter(range(len(reward_list)), reward_list)
    plt.show()

    plt.plot(loss_curve)
    plt.show()


    #test
    state = env.reset()
    test = ActorCritic(env)
    test.load_state_dict(torch.load("./actor_critic_model.pth"))
    total_reward = 0
    while True:
        env.render()
        action = int(test.actor(torch.Tensor(state.reshape(1, -1))).argmax(1)[0])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state