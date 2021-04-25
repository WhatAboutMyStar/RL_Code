from RunningBall_v1 import RunningBall
import torch
import torch.nn.functional as F
from collections import deque
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange

#保存经验 每局游戏会迭代200次，200 * 局数
buffer = deque(maxlen=200*800)
load_model = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'

#agent
class Net(torch.nn.Module):
    def __init__(self, env):
        super(Net, self).__init__()
        self.state_space = 9
        self.action_space = 5
        self.fc1 = torch.nn.Linear(in_features=self.state_space, out_features=128)
        self.fc2 = torch.nn.Linear(in_features=128, out_features=128)
        self.fc3 = torch.nn.Linear(in_features=128, out_features=128)
        self.fc4 = torch.nn.Linear(in_features=128, out_features=self.action_space)


    def forward(self, state):
        x = self.fc1(state)
        x = F.relu(x)
        c = x
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = x + c
        x = F.relu(x)
        x = self.fc4(x)
        # x = F.softmax(x, dim=1)
        return x

if __name__ == '__main__':
    env = RunningBall()

    agent = Net(env)
    agent.to(device)
    agent_target = Net(env)
    agent_target.to(device)
    if load_model:
        agent.load_state_dict(torch.load("./DQN_model.pth"))
    agent_target.load_state_dict(agent.state_dict())

    optimizer = torch.optim.Adam(agent.parameters(), lr=0.0001)
    loss_fn = torch.nn.MSELoss()

    state = env.reset()

    batch_size = 128
    gamma = 0.98
    start_learning = 1000
    epsilon = 0.98

    all_reward = 0
    reward_list = []
    loss_curve = []
    for i in trange(200 * 2000):
        # env.render()
        #选择一个动作
        # if i < start_learning:
        #     action = env.sample()
        if np.random.rand() < epsilon:
            action = int(agent(torch.from_numpy(state).float().unsqueeze(0).to(device)).argmax(1)[0])
        else:
            action = env.sample()

        #与环境交互
        next_state, reward, done, info = env.step(action)
        reward = env.distance(env.x, env.y, env.enemy_x, env.enemy_y) - 50
        reward = reward / 100


        buffer.append([state, action, reward, next_state, done])
        all_reward += reward

        if done:
            reward_list.append(all_reward)
            all_reward = 0
            state = env.reset()
        else:
            state = next_state

        if i >= start_learning and i % 10 == 0:
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

            batch_state = torch.Tensor(batch_state).to(device)
            batch_action = torch.Tensor(batch_action).to(device)
            batch_reward = torch.Tensor(batch_reward).to(device)
            batch_next_state = torch.Tensor(batch_next_state).to(device)
            masks = torch.Tensor(1 - np.array(batch_done) * 1).to(device)

            next_action = agent(batch_next_state).argmax(1)

            Q_value = agent(batch_state)[range(batch_size), np.array(batch_action, dtype=np.int32)]
            Q_target = (batch_reward + masks * gamma * agent_target(batch_next_state)[range(batch_size), next_action])
            loss = loss_fn(Q_target, Q_value)

            agent.zero_grad()
            loss.backward()
            optimizer.step()

            #更新targetNet
            if i % 200 == 0:
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
    total_reward = 0
    while True:
        env.render()
        action = int(testAgent(torch.from_numpy(state).float().unsqueeze(0)).argmax(1)[0])
        next_state, reward, done, info = env.step(action)
        # print(reward)

        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state

