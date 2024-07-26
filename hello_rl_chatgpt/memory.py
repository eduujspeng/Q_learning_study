# 导入必要的库
import numpy as np  # 导入numpy库

# 定义ReplayMemory类
class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity  # 记忆库容量
        self.memory = np.zeros((capacity, 2 * 2 ))  # 初始化记忆库
        self.memory_counter = 0  # 记忆计数器

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))  # 创建转移
        index = self.memory_counter % self.capacity  # 确定存储位置
        self.memory[index, :] = transition  # 存储转移
        self.memory_counter += 1  # 增加记忆计数器

    def sample(self, batch_size):
        sample_index = np.random.choice(self.capacity, batch_size)  # 随机抽取批量记忆
        return self.memory[sample_index, :]  # 返回抽取的记忆
