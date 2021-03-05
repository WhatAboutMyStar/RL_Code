import torch
import torch.nn.functional as F
import gym
import numpy as np

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
    agent.load_state_dict(torch.load("./DQN_model.pth"))

    state = env.reset()
    while True:
        env.render()
        action = int(agent(torch.Tensor(state.reshape(1, -1))).argmax(1)[0])
        next_state, reward, done, info = env.step(action)
        if done:
            break
        state = next_state
