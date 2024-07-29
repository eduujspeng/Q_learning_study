import random
import gym
import torch

# 创建CartPole-v1环境
def make_env():
    env = gym.make('CartPole-v1')
    return env

# 与环境交互的函数
def interact(env, policy_net, epsilon, device):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        # 转换状态到模型需要的格式
        state = torch.tensor(state, dtype=torch.float32).to(device).unsqueeze(0)
        # 使用epsilon-greedy策略选择动作
        if random.random() < epsilon:
            action = env.action_space.sample()  # 随机选择动作
        else:
            # 使用神经网络预测动作
            action = policy_net(state).max(1)[1].item()
        state, reward, done, _ = env.step(action)
        total_reward += reward
    
    return total_reward