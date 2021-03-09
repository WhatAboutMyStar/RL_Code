import torch
import torch.nn as nn
import torch.nn.functional as F
import gym

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
    # env = gym.make("MountainCar-v0")
    env = gym.make("CartPole-v0")
    agent = ActorCritic(env)
    agent.load_state_dict(torch.load("./actor_critic_model.pth"))

    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        action = int(agent.actor(torch.Tensor(state.reshape(1, -1))).argmax(1)[0])
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state