import numpy as np  # 导入NumPy库，用于数组和数值操作
import random  # 导入随机模块
from collections import deque  # 从collections库中导入deque，用于创建双端队列
from models.DQN_model import build_model

class DQNAgent:  # 定义DQN代理类
    def __init__(self, state_size, action_size):  # 初始化方法，接受状态大小和动作大小作为参数
        self.state_size = state_size  # 保存状态大小
        self.action_size = action_size  # 保存动作大小
        self.memory = deque(maxlen=2000)  # 初始化经验回放记忆库，最大长度为2000
        self.gamma = 0.95  # 折扣因子
        self.epsilon = 1.0  # 探索率
        self.epsilon_min = 0.01  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减
        self.model = build_model(state_size, action_size)  # 创建Q网络模型
  

    def remember(self, state, action, reward, next_state, done):  # 记忆方法
        self.memory.append((state, action, reward, next_state, done))  # 将经验存入记忆库

    def act(self, state):  # 动作选择方法
        if np.random.rand() <= self.epsilon:  # 如果随机数小于等于探索率
            return random.randrange(self.action_size)  # 随机选择一个动作
        act_values = self.model(state)  # 使用Q网络预测动作值
        return np.argmax(act_values[0])  # 返回最大动作值对应的动作

    def replay(self, batch_size):  # 经验回放方法
        minibatch = random.sample(self.memory, batch_size)  # 随机采样一个小批量
        for state, action, reward, next_state, done in minibatch:  # 遍历小批量中的每一个经验
            target = reward  # 初始化目标值为奖励
            if not done:  # 如果未结束
                target = reward + self.gamma * np.amax(self.model(next_state)[0])  # 更新目标值
            target_f = self.model(state).numpy()  # 获取当前状态的Q值
            target_f[0][action] = target  # 更新Q值
            self.model.fit(state, target_f, epochs=1, verbose=0)  # 训练模型，更新参数
        if self.epsilon > self.epsilon_min:  # 如果探索率大于最小探索率
            self.epsilon *= self.epsilon_decay  # 衰减探索率
        
