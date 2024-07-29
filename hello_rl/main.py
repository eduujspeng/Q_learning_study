'''
环境是一个一维世界, 在世界的右边有宝藏, 探索者只要得到宝藏尝到了甜头, 然后以后就记住了得到宝藏的方法, 这就是他用强化学习所学习到的行为.
'''
import torch
from env import enviroment
from agent import PolicyNetwork
import torch.optim as optim
from agent import Agent

env = enviroment()

input_dim = 1  # 状态空间维度
output_dim = len(env.action_space)  # 动作空间大小

agent = Agent(input_dim, output_dim, 1e-3, 0.9, 1, 32, 2)

episodes = 10
for i in range(episodes):
    mxz_step = 0
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done = env.step(action)
        agent.store_transition(state, action, next_state, reward)

        state = next_state
        env.render()

        if len(agent.memory) > 32:
            agent.learn()

        mxz_step += 1

    print('\n{}轮奖励为{}, epsilon is {}'.format(i+1, mxz_step, agent.epsilon))
