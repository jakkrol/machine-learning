import torch
from torch import nn
import matplotlib.pyplot as plt

weight = 1.2
bias = 0.5

start = 0
end = 1
step = 0.02

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
        self.l1 = nn.Linear(1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        return x

torch.manual_seed(42)
model = myModel()
print(list(model.parameters()))

with torch.inference_mode():
    preds = model(test_x)

print(f"Number of testing samples: {len(test_x)}") 
print(f"Number of predictions made: {len(preds)}")
print(f"Predicted values:\n{preds}")


loss_fn = nn.L1Loss()
opt = torch.optim.AdamW(params=model.parameters(), lr=0.01)

train_loss_values = []
test_loss_values = []
epoch_count = []

Repoch = 10
for epoch in range(Repoch):
    model.train()
    y_pred = model(train_x)
    loss = loss_fn(y_pred, train_y)
    opt.zero_grad()
    loss.backward()
    opt.step()

    with torch.inference_mode():
        test_pred = model(test_x)
        test_loss = loss_fn(test_pred, test_y.type(torch.float))

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            train_loss_values.append(loss.detach().numpy())
            test_loss_values.append(test_loss.detach().numpy())
            print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")


model.eval()
with torch.inference_mode():
    preds = model(test_x)
print("PREDS")
print(preds)

def plot2(train_x, train_y, test_x, preds):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_x, train_y, c="blue", label="Train data")
    plt.scatter(test_x, preds, c="red", label="Predictions")

    plt.grid(True)
    plt.legend()

plot2(train_x, train_y, test_x, preds)
plt.show()