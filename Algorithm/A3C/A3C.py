import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
from tqdm import trange
import numpy as np
import matplotlib.pyplot as plt

n_train_processes = 3
learning_rate = 0.0001
update_interval = 10
gamma = 0.98
total_game = 1000

load_model = False

class ActorCritic(nn.Module):
    def __init__(self):
        super(ActorCritic, self).__init__()
        self.fc = nn.Linear(in_features=4, out_features=128)
        self.p = nn.Linear(in_features=128, out_features=2)
        self.v = nn.Linear(in_features=128, out_features=1)

    def pi(self, x):
        x = F.relu(self.fc(x))
        x = F.softmax(self.p(x), dim=1)
        return x

    def value(self, x):
        x = F.relu(self.fc(x))
        x = self.v(x)
        return x


def train(global_model):
    local_model = ActorCritic()
    local_model.load_state_dict(global_model.state_dict())

    optimizer = torch.optim.Adam(global_model.parameters(), lr=learning_rate)
    loss_curve = []
    env = gym.make('CartPole-v1')

    for i in trange(total_game):
        done = False
        state = env.reset()
        while not done:
            state_list, action_list, reward_list = [], [], []
            for t in range(update_interval):
                prob = local_model.pi(torch.from_numpy(state.reshape(1, -1)).float())
                m = Categorical(prob)
                action = m.sample().item()
                next_state, reward, done, info = env.step(action)

                state_list.append(state)
                action_list.append([action])
                reward_list.append(reward/100.0)

                state = next_state
                if done:
                    break

            state_final = torch.tensor(next_state, dtype=torch.float)
            R = 0.0 if done else local_model.value(state_final).item()
            td_target_list = []
            for reward in reward_list[::-1]:
                R = gamma * R + reward
                td_target_list.append([R])
            td_target_list.reverse()

            state_batch, action_batch, td_target = torch.tensor(state_list, dtype=torch.float), torch.tensor(action_list), \
                torch.tensor(td_target_list)
            advantage = td_target - local_model.value(state_batch)

            pi = local_model.pi(state_batch)
            pi_a = pi.gather(1, action_batch)
            loss = (-torch.log(pi_a) * advantage.detach() + F.smooth_l1_loss(local_model.value(state_batch), td_target.detach())).mean()
            optimizer.zero_grad()
            loss.backward()
            loss_curve.append(loss)
            for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
                global_param._grad = local_param.grad
            optimizer.step()
            local_model.load_state_dict(global_model.state_dict())

    env.close()
    plt.plot(loss_curve)
    plt.show()



if __name__ == '__main__':
    global_model = ActorCritic()
    if load_model:
        global_model.load_state_dict(torch.load("./agent.pth"))
    global_model.share_memory()

    processes = []
    for i in range(n_train_processes):
        p = mp.Process(target=train, args=(global_model, ))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    torch.save(global_model.state_dict(), "./agent.pth")

    #test
    env = gym.make("CartPole-v1")
    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        prob = global_model.pi(torch.from_numpy(state.reshape(1, -1)).float())
        m = Categorical(prob)
        action = m.sample().item()
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state
