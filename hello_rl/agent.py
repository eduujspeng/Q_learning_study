import numpy as np
import torch
import torch.nn as nn
from torch import optim
from memory import Transition

from memory import ReplayMemory

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 12)
        self.fc2 = nn.Linear(12, output_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs
    
class Agent():
    def __init__(self, n_state, n_action, lr, gamma, epsilon, batch_size, target_replace_iter) -> None:
        self.eval_net = PolicyNetwork(n_state, n_action)  # 评估网络
        self.target_net = PolicyNetwork(n_state, n_action)  # 目标网络
        self.memory = ReplayMemory()  # 记忆库
        self.optimizer = optim.Adam(self.eval_net.parameters(), lr=lr)  # Adam优化器
        self.loss_func = nn.MSELoss()  # 均方误差损失函数
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # 贪婪策略
        self.target_replace_iter = target_replace_iter  # 目标网络更新频率
        self.batch_size = batch_size  # 批量大小
        self.learn_step_counter = 0  # 学习步数计数器

        self.min_epsilon = 0.1
        self.espilon_iter = 0.99

        self.n_action = n_action

    def select_action(self, state):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), dim=0)  # 扩维
        if np.random.uniform() > self.epsilon:
            action_value = self.eval_net(state)
            action = torch.argmax(action_value).item()
        else:
            action = np.random.randint(0, self.n_action)
        return action

    def store_transition(self, s, a, s_, r):
        self.memory.push(s, a, s_, r)

    def learn(self):
        if self.learn_step_counter % self.target_replace_iter == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        b_memory = self.memory.sample(self.batch_size)

        batch = Transition(*zip(*b_memory))

        state = torch.tensor(np.array(batch.state), dtype=torch.float)
        rewards = torch.tensor(batch.reward, dtype=torch.int64)
        state_ = torch.tensor(np.array(batch.next_state), dtype=torch.float)
        action = torch.tensor(batch.action, dtype=torch.int64)

        q_eval = self.eval_net(state).gather(1, action.unsqueeze(1)).view(self.batch_size, )
        # print(q_eval.shape)

        q_next = self.target_net(state_).detach()    
        q_next = q_next.max(1)[0].view(self.batch_size, )   
        # print(q_next.shape)

        q_target = rewards + self.gamma * q_next  # Q目标值

        loss = self.loss_func(q_eval, q_target)  # 计算损失
        self.optimizer.zero_grad()  # 梯度清零
        loss.backward()  # 反向传播
        self.optimizer.step()  # 优化更新 

        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.espilon_iter

