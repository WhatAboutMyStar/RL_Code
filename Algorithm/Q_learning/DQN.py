import torch
import torch.nn.functional as F
from collections import deque
import gym
import numpy as np
import matplotlib.pyplot as plt

#保存经验
buffer = deque(maxlen=100000)

#agent
class Net(torch.nn.Module):
    def __init__(self, env):
        super(Net, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        self.fc1 = torch.nn.Linear(in_features=self.state_space, out_features=200)
        self.fc2 = torch.nn.Linear(in_features=200, out_features=self.action_space)


    def forward(self, state):
        x = self.fc1(state)
        x = F.softplus(x)
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    env = gym.make("MountainCar-v0")

    agent = Net(env)
    agent_target = Net(env)
    agent_target.load_state_dict(agent.state_dict())

    optimizer = torch.optim.SGD(agent.parameters(), lr=0.001)
    loss_fn = torch.nn.MSELoss()

    state = env.reset()

    batch_size = 128
    gamma = 0.9
    start_learning = 1000

    all_reward = 0
    reward_list = []
    loss_curve = []
    for i in range(200000):
        #选择一个动作
        if i < start_learning:
            action = env.action_space.sample()
        elif np.random.rand() < 0.9:
            action = int(agent(torch.Tensor(state.reshape(1, -1))).argmax(1)[0])
        else:
            action = env.action_space.sample()

        #与环境交互
        next_state, reward, done, info = env.step(action)

        buffer.append([state, action, reward, next_state, done])
        all_reward += reward

        if done:
            reward_list.append(all_reward)
            all_reward = 0
            state = env.reset()
        else:
            state = next_state

        if i >= start_learning and i % 4 == 0:
            sample_index = np.random.choice(len(buffer), batch_size, replace=False)

            batch_state = []
            batch_action = []
            batch_reward = []
            batch_next_state = []
            batch_done = []

            #经验回放
            for index in sample_index:
                sample = buffer[index]

                batch_state.append(sample[0])
                batch_action.append(sample[1])
                batch_reward.append(sample[2])
                batch_next_state.append(sample[3])
                batch_done.append(sample[4])

            batch_state = torch.Tensor(batch_state)
            batch_action = torch.Tensor(batch_action)
            batch_reward = torch.Tensor(batch_reward)
            batch_next_state = torch.Tensor(batch_next_state)
            masks = torch.Tensor(1 - np.array(batch_done) * 1)

            next_action = agent(batch_next_state).argmax(1)

            Q_value = agent(batch_state)[range(batch_size), np.array(batch_action, dtype=np.int32)]
            Q_target = (batch_reward + gamma * agent_target(batch_next_state)[range(batch_size), next_action])
            loss = loss_fn(Q_value, Q_target)

            agent.zero_grad()
            loss.backward()
            optimizer.step()

            #更新targetNet
            if i % 1000 == 0:
                agent_target.load_state_dict(agent.state_dict())

            loss_curve.append(loss)

    torch.save(agent.state_dict(), "./DQN_model.pth")

    plt.scatter(range(len(reward_list)), reward_list)
    plt.show()

    plt.plot(loss_curve)
    plt.show()

    #test
    testAgent = Net(env)
    testAgent.load_state_dict(torch.load("./DQN_model.pth"), strict=False)
    state = env.reset()
    while True:
        env.render()
        action = int(testAgent(torch.Tensor(state.reshape(1, -1))).argmax(1)[0])
        next_state, reward, done, info = env.step(action)
        if done:
            break
        state = next_state