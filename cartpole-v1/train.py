import gym
import numpy as np
from agent import DQNAgent

env = gym.make('CartPole-v1')  # 创建CartPole-v1环境
state_size = env.observation_space.shape[0]  # 获取状态空间的大小
action_size = env.action_space.n  # 获取动作空间的大小
agent = DQNAgent(state_size, action_size)  # 创建DQN代理
episodes = 1000  # 定义训练的回合数

for e in range(episodes):  # 循环训练每一个回合
    state,_ = env.reset()  # 重置环境，获取初始状态
    state = np.reshape(state, [1, state_size])  # 调整状态的形状
    for time in range(500):  # 在每个回合中执行最多500步
        action = agent.act(state)  # 选择一个动作
        next_state, reward, done, _, _ = env.step(action)  # 执行动作，获取下一状态、奖励和结束标志
        reward = reward if not done else -10  # 如果未结束，奖励保持不变，否则设为-10
        next_state = np.reshape(next_state, [1, state_size])  # 调整下一状态的形状
        agent.remember(state, action, reward, next_state, done)  # 存储经验
        state = next_state  # 更新当前状态
        if done:  # 如果回合结束
            print(f"episode: {e}/{episodes}, score: {time}, e: {agent.epsilon:.2}")  # 打印回合信息
            break  # 跳出循环
        agent.replay()  # 经验回放
