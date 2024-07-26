'''
环境是一个一维世界, 在世界的右边有宝藏, 探索者只要得到宝藏尝到了甜头, 然后以后就记住了得到宝藏的方法, 这就是他用强化学习所学习到的行为.
Q-learning 是一种记录行为值 (Q value) 的方法, 每种在一定状态的行为都会有一个值 Q(s, a), 就是说 行为 a 在 s 状态的值是 Q(s, a). s 在上面的探索者游戏中, 就是 o 所在的地点了. 
而每一个地点探索者都能做出两个行为 left/right, 这就是探索者的所有可行的 a 啦.
'''
# 导入必要的库
import time  # 导入time库
import torch  # 导入torch库
import numpy as np  # 导入numpy库
from environment import Environment  # 导入环境类
from DQN import DQN  # 导入DQN类

# 设置随机种子以确保结果可重复
np.random.seed(2)
torch.manual_seed(2)

# 超参数
N_STATES = 6   # 一维世界的长度
ACTIONS = ['left', 'right']  # 可用的动作
EPSILON = 0.9  # 贪婪策略
GAMMA = 0.9  # 折扣因子
LR = 0.01  # 学习率
BATCH_SIZE = 32  # 批量大小
MEMORY_CAPACITY = 200  # 记忆库容量
TARGET_REPLACE_ITER = 100  # 目标网络更新频率
MAX_EPISODES = 13  # 最大回合数
FRESH_TIME = 0.3  # 每次移动的刷新时间

# 初始化环境和DQN
env = Environment(N_STATES)
dqn = DQN(1, len(ACTIONS), LR, GAMMA, EPSILON, MEMORY_CAPACITY, BATCH_SIZE, TARGET_REPLACE_ITER)

# 强化学习主循环
for episode in range(MAX_EPISODES):
    state = env.reset()  # 初始化状态
    step_counter = 0  # 步数计数器
    env.render(step_counter, episode)  # 渲染环境
    while True:
        action = dqn.choose_action(state)  # 选择动作
        next_state, reward, done = env.step(action)  # 执行动作并获得下一个状态和奖励
        dqn.store_transition(state, action, reward, next_state)  # 存储记忆
        state = next_state  # 更新当前状态
        env.render(step_counter, episode)  # 渲染环境
        if dqn.memory_counter > MEMORY_CAPACITY:  # 当记忆库满时开始学习
            dqn.learn()
        if done:  # 如果到达终点则结束当前回合
            # print('冒险者一共走了{}步'.format(step_counter))
            break
            
        step_counter += 1
        

print('\n训练结束！')
