'''
环境是一个一维世界, 在世界的右边有宝藏, 探索者只要得到宝藏尝到了甜头, 然后以后就记住了得到宝藏的方法, 这就是他用强化学习所学习到的行为.
'''
import torch
from env import env
from agent import PolicyNetwork
import torch.optim as optim

env = env()

input_dim = env.N_STATES  # 状态空间维度
output_dim = env.n_action  # 动作空间大小

policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

def evaluate_policy(policy_net, env, episodes=10):
    for i in range(episodes):
        mxz_step = 0
        state = env.reset()
        done = False
        
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            next_state, reward, done = env.step(action)
            state = next_state

            mxz_step += 1

        print('{}轮奖励为{}'.format(i, mxz_step))


# 使用上文定义的PolicyNetwork和初始化的env
episodes = 100
average_reward = evaluate_policy(policy_net, env, episodes=episodes)
print(f"Average reward over {episodes} episodes: {average_reward}")
