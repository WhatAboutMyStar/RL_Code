from A2C import *

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    total_reward = 0
    agent = ActorCritic(env)
    agent.load_state_dict(torch.load("./agent.pth"))
    state = env.reset()
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

