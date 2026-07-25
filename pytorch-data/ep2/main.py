import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 1.2
bias = 0.5

start = 0
end = 1
step = 0.1

X = torch.arange(start, end, step).unsqueeze(1)
y = weight * X + bias

def plot(data=X, data2=y):
    plt.figure(figsize=(10, 7))
    plt.scatter(data, data2, c="g")
    plt.grid(True)

# plot()
# plt.show()

train_split = int(0.8 * len(X))
train_x, train_y = X[:train_split], y[:train_split]
test_x, test_y = X[train_split:], y[train_split:]

class myModel(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x