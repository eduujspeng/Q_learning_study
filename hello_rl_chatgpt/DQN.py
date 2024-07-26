# 导入必要的库
import numpy as np  # 导入numpy库
import torch  # 导入torch库
import torch.nn as nn  # 导入torch.nn模块
import torch.nn.functional as F  # 导入torch.nn.functional模块
import torch.optim as optim  # 导入torch.optim模块
from memory import ReplayMemory  # 导入ReplayMemory类

# 定义神经网络
class Net(nn.Module):
    def __init__(self, n_states, n_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_states, 4)  # 第一层全连接层
        self.fc1.weight.data.normal_(0, 0.1)  # 初始化权重
        self.out = nn.Linear(4, n_actions)  # 输出层
        self.out.weight.data.normal_(0, 0.1)  # 初始化权重

    def forward(self, x):
        x = F.relu(self.fc1(x))  # ReLU激活函数
        actions_value = self.out(x)  # 输出动作值
        return actions_value

# 定义DQN类
class DQN:
    def __init__(self, n_states, n_actions, lr, gamma, epsilon, memory_capacity, batch_size, target_replace_iter):
        self.eval_net = Net(n_states, n_actions)  # 评估网络
        self.target_net = Net(n_states, n_actions)  # 目标网络
        self.memory = ReplayMemory(memory_capacity)  # 记忆库
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)  # Adam优化器
        self.loss_func = nn.MSELoss()  # 均方误差损失函数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 贪婪策略
        self.target_replace_iter = target_replace_iter  # 目标网络更新频率
        self.batch_size = batch_size  # 批量大小
        self.memory_counter = 0  # 记忆计数器
        self.learn_step_counter = 0  # 学习步数计数器

    def choose_action(self, state):
        state = torch.unsqueeze(torch.FloatTensor(state), 0)  # 将状态转换为张量
        if np.random.uniform() < self.epsilon:  # 选择贪婪策略
            actions_value = self.eval_net.forward(state)
            action = torch.max(actions_value, 1)[1].data.numpy()  # 选择Q值最大的动作
            action = action[0]
        else:  # 随机选择动作
            action = np.random.randint(0, self.eval_net.out.out_features)
        return action

    def store_transition(self, s, a, r, s_):
        self.memory.store_transition(s, a, r, s_)  # 存储记忆
        self.memory_counter += 1  # 增加记忆计数器

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:  # 更新目标网络
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_memory = self.memory.sample(self.batch_size)  # 随机抽取批量记忆
        b_s = torch.FloatTensor(b_memory[:, :self.eval_net.fc1.in_features])
        b_a = torch.LongTensor(b_memory[:, self.eval_net.fc1.in_features:self.eval_net.fc1.in_features + 1].astype(int))
        b_r = torch.FloatTensor(b_memory[:, self.eval_net.fc1.in_features + 1:self.eval_net.fc1.in_features + 2])
        b_s_ = torch.FloatTensor(b_memory[:, -self.eval_net.fc1.in_features:])

        q_eval = self.eval_net(b_s).gather(1, b_a)  # 评估网络Q值
        q_next = self.target_net(b_s_).detach()  # 目标网络Q值
        q_target = b_r + self.gamma * q_next.max(1)[0].view(self.batch_size, 1)  # Q目标值

        loss = self.loss_func(q_eval, q_target)  # 计算损失
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        self.optimizer.step()  # 优化更新
