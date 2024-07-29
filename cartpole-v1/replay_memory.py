import random
from collections import namedtuple, deque

# 定义经验元组
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# 经验回放类
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)
    
    def push(self, *args):
        """添加经验到回放中"""
        self.memory.append(Transition(*args))
    
    def sample(self, batch_size):
        """随机采样经验"""
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)