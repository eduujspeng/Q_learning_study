import gym  # 导入OpenAI Gym库，用于创建和交互环境
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
import torch.nn.functional as F  # 导入PyTorch的功能模块
import numpy as np  # 导入NumPy库，用于数组和数值操作
import random  # 导入随机模块
from collections import deque  # 从collections库中导入deque，用于创建双端队列
from DQN import QNetwork

class DQNAgent:  # 定义DQN代理类
    def __init__(self, state_size, action_size):  # 初始化方法，接受状态大小和动作大小作为参数
        self.state_size = state_size  # 保存状态大小
        self.action_size = action_size  # 保存动作大小
        self.memory = deque(maxlen=2000)  # 初始化经验回放记忆库，最大长度为2000
        self.gamma = 0.99  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.learning_rate = 0.001  # 学习率
        self.batch_size = 64  # 批量大小
        self.model = QNetwork(state_size, action_size)  # 创建Q网络
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)  # 创建Adam优化器
        self.criterion = nn.MSELoss()  # 定义损失函数为均方误差

    def remember(self, state, action, reward, next_state, done):  # 记忆方法
        self.memory.append((state, action, reward, next_state, done))  # 将经验存入记忆库

    def act(self, state):  # 动作选择方法
        if np.random.rand() <= self.epsilon:  # 如果随机数小于等于探索率
            return random.choice(range(self.action_size))  # 随机选择一个动作
        state = torch.FloatTensor(state)  # 将状态转换为Tensor，并增加一个维度
        act_values = self.model(state)  # 使用Q网络预测动作值
        return torch.argmax(act_values, dim=1).item()  # 返回最大动作值对应的动作

    def replay(self):  # 经验回放方法
        if len(self.memory) < self.batch_size:  # 如果记忆库中的经验数量小于批量大小
            return  # 直接返回
        minibatch = random.sample(self.memory, self.batch_size)  # 随机采样一个小批量
        for state, action, reward, next_state, done in minibatch:  # 遍历小批量中的每一个经验
            state = torch.FloatTensor(state)  # 将状态转换为Tensor
            next_state = torch.FloatTensor(next_state)  # 将下一状态转换为Tensor
            target = reward  # 初始化目标值为奖励
            if not done:  # 如果未结束
                target = reward + self.gamma * torch.max(self.model(next_state).detach())  # 更新目标值
            target_f = self.model(state)[0]  # 获取当前状态的Q值
            target_f[action] = target  # 更新Q值
            self.optimizer.zero_grad()  # 梯度清零
            loss = self.criterion(self.model(state).squeeze(0), target_f)  # 计算损失
            loss.backward()  # 反向传播
            self.optimizer.step()  # 更新参数
        if self.epsilon > self.epsilon_min:  # 如果探索率大于最小探索率
            self.epsilon *= self.epsilon_decay  # 衰减探索率
