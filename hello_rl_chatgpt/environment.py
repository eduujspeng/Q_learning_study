# 导入必要的库
import time  # 导入time库
import numpy as np  # 导入numpy库

# 定义环境类
class Environment:
    def __init__(self, n_states):
        self.n_states = n_states  # 状态数目
        self.actions = ['left', 'right']  # 动作
        self.terminal_state = n_states - 1  # 终点状态
        self.state = 0  # 初始化状态

    def reset(self):
        self.state = 0  # 重置状态
        return np.array([self.state])  # 返回初始状态

    def step(self, action):
        state = self.state  # 当前状态
        if action == 1:  # 向右移动
            if state == self.n_states - 2:  # 到达终点
                next_state = self.terminal_state
                reward = 1
                done = True
            else:
                next_state = state + 1
                reward = 0
                done = False
        else:  # 向左移动
            reward = 0
            if state == 0:
                next_state = state  # 到达墙壁
            else:
                next_state = state - 1
            done = False

        self.state = next_state  # 更新状态
        return np.array([next_state]), reward, done  # 返回新的状态，奖励和结束标志

    def render(self, step_counter, episode):
        env_list = ['-'] * (self.n_states - 1) + ['T']  # '---------T' 表示环境
        if self.state == self.terminal_state:
            interaction = 'Episode %s: total_steps = %s' % (episode + 1, step_counter)
            print('\r{}'.format(interaction), end='')
            time.sleep(2)
            print('\r                                ', end='')
        else:
            env_list[self.state] = 'o'
            interaction = ''.join(env_list)
            print('\r{}'.format(interaction), end='')
            time.sleep(0.3)
