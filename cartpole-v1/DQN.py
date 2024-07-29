import gym  # 导入OpenAI Gym库，用于创建和交互环境
import torch  # 导入PyTorch库
import torch.nn as nn  # 导入PyTorch的神经网络模块
import torch.optim as optim  # 导入PyTorch的优化器模块
import torch.nn.functional as F  # 导入PyTorch的功能模块
import numpy as np  # 导入NumPy库，用于数组和数值操作
import random  # 导入随机模块
from collections import deque  # 从collections库中导入deque，用于创建双端队列

class QNetwork(nn.Module):  # 定义Q网络类，继承自nn.Module
    def __init__(self, state_size, action_size):  # 初始化方法，接受状态大小和动作大小作为参数
        super(QNetwork, self).__init__()  # 调用父类的初始化方法
        self.fc1 = nn.Linear(state_size, 64)  # 定义第一个全连接层
        self.fc2 = nn.Linear(64, 64)  # 定义第二个全连接层
        self.fc3 = nn.Linear(64, action_size)  # 定义第三个全连接层

    def forward(self, state):  # 定义前向传播方法
        x = F.relu(self.fc1(state))  # 使用ReLU激活函数处理第一层输出
        x = F.relu(self.fc2(x))  # 使用ReLU激活函数处理第二层输出
        return self.fc3(x)  # 返回第三层输出，不使用激活函数
