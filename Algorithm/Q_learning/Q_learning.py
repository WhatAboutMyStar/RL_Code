import gym
import numpy as np
import matplotlib.pyplot as plt
# FrozenLake-v0是一个4*4的网络格子，每个格子可以是起始块，目标块、冻结块或者危险块。
# 我们的目标是让智能体学习如何从开始块如何行动到目标块上，而不是移动到危险块上。
# 智能体可以选择向上、向下、向左或者向右移动，同时游戏中还有可能吹来一阵风，将智能体吹到任意的方块上。

env = gym.make("FrozenLake-v0")

#建立并初始化Q表 shape是状态空间的大小和动作空间的大小
Q_table = np.zeros([env.observation_space.n, env.action_space.n])

#超参数设置
lr = 0.85
gamma = 0.99


reward_list = []
render = False

for i in range(10000):
    state = env.reset()
    all_reward = 0

    for j in range(100):
        if render:
            env.render()

        #选择一个动作，并做出这个动作与环境交互得到反馈，添加了噪声
        action = np.argmax(Q_table[state, :] + np.random.randn(1, env.action_space.n) * (1 * (i + 1)))
        next_state, reward, done, info = env.step(action)

        reward = reward - 0.01
        #更新Q表
        Q_table[state, action] = Q_table[state, action] + lr * (reward + gamma * np.max(Q_table[next_state, :]) - Q_table[state, action])

        all_reward += reward
        state = next_state
        if done:
            break

    reward_list.append(all_reward)

print(Q_table)
plt.scatter(range(len(reward_list)), reward_list)
plt.show()
