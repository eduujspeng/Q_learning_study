import numpy as np
import gym
from gym import spaces
from utils.LCMOperation import *
from utils.constants import EPOCHS, DATA_HEADER, AVAILABLE_DATA_HEADER, TIME_STEP, LCM_CONTROL_INTERVAL

class TemperatureControlEnv(gym.Env):
    def __init__(self):
        super(TemperatureControlEnv, self).__init__()
        
        # 定义状态空间，包含'通道温度', '电池温度', '目标温度'
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([100, 100, 100]), dtype=np.float32)
        
        # 定义动作空间，包含PWM值，从0到100的离散值
        self.action_space = spaces.Discrete(101)
        
        # 初始化状态
        self.state = np.array([50, 50, 50], dtype=np.float32)
        self.target_temperature_reached = False

        # 初始化客户端
        self.client = create_client()

    def step(self, action):
        # 根据action更新状态，这里简单模拟
        # self.state[0] += (action - 50) * 0.1  # 通道温度变化
        # self.state[1] += (action - 50) * 0.05  # 电池温度变化
        operate_b101(self.client, action)
        self.state, pwm = self._get_info()
        self.state = np.clip(self.state, 0, 100)
        done = False
        
        # 计算reward，电池温度第一次达到目标温度后开始奖励，越接近越好
        if not self.target_temperature_reached and (abs(self.state[1] - self.state[2]) < 1 or self.state[1] >= self.state[2]):
            self.target_temperature_reached = True

        if self.target_temperature_reached:
            reward = 1-abs(self.state[1] - self.state[2])
            # 判断是否结束
            done = abs(self.state[1] - self.state[2]) > 1  # 电池温度超过1度则训练停止
        else:
            reward = 1-abs(self.state[1] - self.state[2])

        return self.state, reward, done, {'pwm':pwm}

    def reset(self):
        # 重置环境状态
        # self.state = np.array([50, 50, 50], dtype=np.float32)
        self.state, pwm = self._get_info()

        self.target_temperature_reached = False
        return self.state, {'target_temperature_reached':self.target_temperature_reached, 'pwm':pwm}
    
    def _get_info(self):
        # 从操作 A103 获取实际数据
        real_data = operate_a103(self.client)
        # 获取 PWM 数据，如果 pwm 为空则获取
        pwm = operate_b103(self.client)

        single_input = []
        if real_data is not None and pwm is not None:
                # 遍历所有可用数据头
                for header in AVAILABLE_DATA_HEADER:
                    if header != 'PWM':
                        # 获取数据头的索引
                        index = DATA_HEADER.index(header)
                        # 添加实际数据到单个输入列表
                        single_input.append(real_data[index])
        self.state = np.array(single_input, dtype=np.float32)
        return self.state, pwm


  
if __name__ == '__main__':
    # 创建环境和DQN网络
    env = TemperatureControlEnv()
    action_size = env.action_space.n
    state_size = env.observation_space.shape[0]

    print(action_size, state_size)
    print("action is {}".format(env.action_space))
    print('state_size is {}'.format(env.state))
    
    import random
    for i in range(10):
        print(random.randrange(action_size))
