import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import torch.multiprocessing as mp
import numpy as np
from tqdm import trange
import matplotlib.pyplot as plt

num_train_processes = 3
lr = 0.0001
gamma = 0.98
total_update = 50000
update_interval = 5
load_model = False
loss_curve = []

class ActorCritic(nn.Module):
    def __init__(self, env):
        super(ActorCritic, self).__init__()
        self.fc1 = nn.Linear(in_features=env.observation_space.shape[0], out_features=128)
        self.actor = nn.Linear(in_features=128, out_features=2)
        self.critic = nn.Linear(in_features=128, out_features=1)

    def pi(self, x, softmax_dim=1):
        x = F.relu(self.fc1(x))
        x = self.actor(x)
        prob = F.softmax(x, dim=softmax_dim)
        return prob

    def value(self, x):
        x = F.relu(self.fc1(x))
        v = self.critic(x)
        return v

def worker(worker_id, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env = gym.make('CartPole-v1')
    env.seed(worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            worker_end.send(ob)
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_spaces':
            worker_end.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class ParallelEnv:
    def __init__(self, n_train_processes):
        self.nenvs = n_train_processes
        self.waiting = False
        self.closed = False
        self.workers = []

        master_ends, worker_ends = zip(*[mp.Pipe() for _ in range(self.nenvs)])
        self.master_ends, self.worker_ends = master_ends, worker_ends

        for worker_id, (master_end, worker_end) in enumerate(zip(master_ends, worker_ends)):
            p = mp.Process(target=worker,
                           args=(worker_id, master_end, worker_end))
            p.daemon = True
            p.start()
            self.workers.append(p)

        # Forbid master to use the worker end for messaging
        for worker_end in worker_ends:
            worker_end.close()

    def step_async(self, actions):
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        return np.stack([master_end.recv() for master_end in self.master_ends])

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def close(self):  # For clean up resources
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
            self.closed = True

def compute_target(value_final, reward_list, mask_list):
    G = value_final.reshape(-1)
    td_target = []

    for r, mask in zip(reward_list[::-1], mask_list[::-1]):
        G = r + gamma * G * mask
        td_target.append(G)

    return torch.tensor(td_target[::-1]).float()

if __name__ == '__main__':
    envs = ParallelEnv(num_train_processes)
    agent = ActorCritic(env=gym.make("CartPole-v1"))
    optimizer = torch.optim.Adam(agent.parameters(), lr=lr)

    if load_model:
        agent.load_state_dict(torch.load("./agent.pth"))

    state = envs.reset()
    for i in trange(total_update):
        state_list = []
        action_list = []
        reward_list = []
        mask_list = []

        for j in range(update_interval):
            prob = agent.pi(torch.from_numpy(state).float())
            action = Categorical(prob).sample().numpy()
            next_state, reward, done, info = envs.step(action)
            state_list.append(state)
            action_list.append(action)
            reward_list.append(reward/100.0)
            mask_list.append(1 - done)

            state = next_state

        state_final = torch.from_numpy(state).float()
        value_final = agent.value(state_final).detach().clone().numpy()
        td_target = compute_target(value_final, reward_list, mask_list)

        td_target_vec = td_target.reshape(-1)
        state_vec = torch.tensor(state_list).float().reshape(-1, 4)
        action_vec = torch.tensor(action_list).reshape(-1, 1)
        advantage = td_target_vec - agent.value(state_vec).reshape(-1)

        pi = agent.pi(state_vec, softmax_dim=1)
        pi_action = pi.gather(1, action_vec).reshape(-1)
        loss = -(torch.log(pi_action) * advantage.detach()).mean() + \
               F.smooth_l1_loss(agent.value(state_vec).reshape(-1), td_target_vec)

        loss_curve.append(loss)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    envs.close()

    plt.plot(loss_curve)
    plt.show()

    torch.save(agent.state_dict(), "./agent.pth")

    #test
    env = gym.make("CartPole-v1")
    state = env.reset()
    total_reward = 0

    while True:
        env.render()
        action_prob = agent.pi(torch.from_numpy(state.reshape(1, -1)).float())
        action = Categorical(action_prob).sample()
        next_state, reward, done, info = env.step(action.item())

        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state
