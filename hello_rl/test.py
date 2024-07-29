import numpy as np
import torch

a= [[1, 0, 0, 0]]
b = [0]
print(a[b[0]])

print(type(np.random.randint(0,1)))

print(torch.argmax(torch.tensor(a)).item())
print(torch.tensor(a).max(1)[1].item())