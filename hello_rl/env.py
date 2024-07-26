import numpy as np  # 导入numpy库
import pandas as pd  # 导入pandas库
import time  # 导入time库

class env():
    def __init__(self) -> None:
        self.action_space = ['l', 'r']
        self.n_action = len(self.action_space)
        self.N_STATES = 6
        self.mxz = 0
        self._build_env()

    def _build_env(self):
        self.env_list = ['-']*(self.N_STATES-1) + ['T']   # '---------T' 表示环境
        self.env_list[self.mxz] = 'o'

    def reset(self):
        self.mxz = 0
        self.env_list = ['-']*(self.N_STATES-1) + ['T']   # '---------T' 表示环境
        self.env_list[self.mxz] = 'o'
        return self.mxz
        
    def step(self, action):
        done = False
        if action == 'right':    # 向右移动
            if self.mxz == self.N_STATES - 2:   # 到达终点
                S_ = 'terminal'
                R = 1
                done = True
            else:
                S_ = self.mxz + 1
                R = 0
        else:   # 向左移动
            R = 0
            if self.mxz == 0:
                S_ = self.mxz  # 到达墙壁
            else:
                S_ = self.mxz - 1
        return S_, R, done

    def render(self):
        time.sleep(0.1)
        print('\r{}'.format(''.join(self.env_list)), end='')