from collections import deque, namedtuple
import random

import torch

# 定义经验元组
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory():
    def __init__(self, maxlen = 1000) -> None:
        self.memory = deque([], maxlen=maxlen)
        self.transition = Transition
    
    def push(self, *args):
        self.memory.append(self.transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
if __name__ == '__main__':
    replay_memory = ReplayMemory()

    replay_memory.push(1, 2, 3, 4)
    replay_memory.push(1, 3, 3, 4)
    replay_memory.push(1, 3, 3, 4)
    replay_memory.push(2, 3, 3, 4)

    print(replay_memory.sample(2))

    transitions = replay_memory.sample(2)
    for t in transitions:
        print("*"*30)
        print(t)
        print(t.state)
    '''
        list(zip('abcdefg', range(3), range(4)))
        [('a', 0, 0), ('b', 1, 1), ('c', 2, 2)]
    '''
    batch = Transition(*zip(*transitions))
    print(batch)
    print(list(zip(*batch)))
    print(*zip(*batch))
    
    print(batch.state, type(batch.state))
    print(torch.tensor(batch.state), torch.tensor(batch.state).shape)

