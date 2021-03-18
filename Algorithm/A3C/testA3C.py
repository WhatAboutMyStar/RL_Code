from A3C import *

if __name__ == '__main__':
    env = gym.make("CartPole-v1")
    agent = ActorCritic()
    agent.load_state_dict(torch.load("./agent.pth"))
    state = env.reset()
    total_reward = 0
    while True:
        env.render()
        prob = agent.pi(torch.from_numpy(state.reshape(1, -1)).float())
        m = Categorical(prob)
        action = m.sample().item()
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state





