from PPO import *
import gym

if __name__ == '__main__':
    env = gym.make(env_name)
    agent = PPO(env, lr, batch_size, gamma)
    agent.load()
    total_reward = 0
    state = env.reset()

    while True:
        env.render()
        action = agent.choose_action(state)
        # action = env.action_space.sample()
        next_state, reward, done, info = env.step(action)

        total_reward += reward
        if done:
            print(total_reward)
            break
        state = next_state