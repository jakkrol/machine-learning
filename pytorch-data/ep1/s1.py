import torch
from torch import nn
import matplotlib.pyplot as plt

print(torch.__version__)
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Used: {device}" )

# Create *known* parameters
weight = 0.7
bias = 0.3

# Create data
start = 0
end = 1
step = 0.02
X = torch.arange(start, end, step).unsqueeze(dim=1)
y = weight * X + bias

#print(X[:10], y[:10])

train_split = int(0.8 * len(X))
X_train, y_train = X[:train_split], y[:train_split]
X_test, y_test = X[train_split:], y[train_split:]

def plot_pr(train_data=X_train,
            train_labels=y_train,
            test_data=X_test,
            test_labels=y_test,
            predictions=None):
    
    plt.figure(figsize=(10,7))
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")

    if predictions is not None:
        plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    
    plt.legend(prop={"size": 14});

#plot_pr()
#plt.show()


class LinearRegressionModel(nn.Module): 
    def __init__(self):
        super().__init__() 
        #self.l1 = nn.Linear(in_features=1, out_features=8)
        #self.l2 = nn.Linear(in_features=8, out_features=1)
        self.l3 = nn.Linear(in_features=1, out_features=1)
        #self.relu = nn.ReLU()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = self.l1(x)
        #x = self.relu(x)
        #x = self.l2(x)
        #x = self.relu(x)
        x = self.l3(x) 
        return x

torch.manual_seed(42)
model_0 = LinearRegressionModel()
print(list(model_0.parameters()))


print(next(model_0.parameters()).device)
model_0.to(device=device)
print(next(model_0.parameters()).device)

X_train = X_train.to(device)
X_test = X_test.to(device)
y_train = y_train.to(device)
y_test = y_test.to(device)

with torch.inference_mode():
    y_preds = model_0(X_test)

# Check the predictions
print(f"Number of testing samples: {len(X_test)}") 
print(f"Number of predictions made: {len(y_preds)}")
print(f"Predicted values:\n{y_preds}")

#plot_pr(predictions=y_preds)
#plt.show()


lf = nn.L1Loss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.01)

# epoch_count = []
# train_lossV = []
# test_lossV = []
epochs = 1000

for epoch in range(epochs):
    model_0.train()
    y_preds = model_0(X_train)
    loss = lf(y_preds, y_train)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


    model_0.eval()

    with torch.inference_mode():
        test_pred = model_0(X_test)
        test_loss = lf(test_pred, y_test.type(torch.float))
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | MAE Train Loss: {loss} | MAE Test Loss: {test_loss} ")

# plt.plot(epoch_count, train_lossV, label="Train loss")
# plt.plot(epoch_count, test_lossV, label="Test loss")
# plt.xlabel("Loss")
# plt.ylabel("Epochs")
# plt.show()

print(model_0.state_dict())
print(f"weights: {weight}, bias: {bias}")


with torch.inference_mode():
    y_preds = model_0(X_test)
plot_pr(predictions=y_preds.cpu())
plt.show()
