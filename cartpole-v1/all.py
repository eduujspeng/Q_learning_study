import gym
import torch
import torch.nn as nn
import torch.optim as optim

env = gym.make('CartPole-v1')

class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, output_dim)
    
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        action_probs = torch.softmax(self.fc2(x), dim=-1)
        return action_probs

input_dim = env.observation_space.shape[0]  # 状态空间维度
output_dim = env.action_space.n  # 动作空间大小

policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

def evaluate_policy(policy_net, env, episodes=10):
    total_rewards = 0
    for i in range(episodes):
        state,_ = env.reset()
        done = False
        episode_reward = 0
        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action_probs = policy_net(state_tensor)
            action = torch.argmax(action_probs).item()
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            state = next_state
        print('{}轮奖励为{}'.format(i, episode_reward))
        total_rewards += episode_reward

    average_reward = total_rewards / episodes
    return average_reward

# 使用上文定义的PolicyNetwork和初始化的env
episodes = 100
average_reward = evaluate_policy(policy_net, env, episodes=episodes)
print(f"Average reward over {episodes} episodes: {average_reward}")
