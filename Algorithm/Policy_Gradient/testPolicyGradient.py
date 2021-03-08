import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

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
    agent = Net(env)
    agent.load_state_dict(torch.load("./pg_model.pth"))

    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = int(agent(torch.Tensor(state.reshape(1, -1))).argmax(1)[0])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state