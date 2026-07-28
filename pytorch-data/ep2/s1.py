import torch
from torch import nn
from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

n_samples = 1000
X, y = make_circles(n_samples=n_samples, noise=0.03, random_state=42)

print(X[:5])

circles = pd.DataFrame({"X1" : X[:, 0], "X2" : X[:, 1], "label": y})
circles.head(10)

# plt.scatter(x=X[:, 0], y=X[:, 1], c=y)
# plt.show()

# train_split = int(0.8 * len(circles))
# print(train_split)

# x_train, y_train = X[:train_split], y[:train_split]
# x_test, y_test = X[train_split:], y[train_split:]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(len(x_train), len(x_test), len(y_train), len(y_test))

class myModel(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        return x