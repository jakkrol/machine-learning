import torch
from torch import nn
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

# Setup training data
train_data = datasets.FashionMNIST(
    root="data", # where to download data to?
    train=True, # get training data
    download=True, # download data if it doesn't exist on disk
    transform=ToTensor(), # images come as PIL format, we want to turn into Torch tensors
    target_transform=None # you can transform labels as well
)

# Setup testing data
test_data = datasets.FashionMNIST(
    root="data",
    train=False, # get test data
    download=True,
    transform=ToTensor()
)

image, label = train_data[0]
print(f"Image shape: {image.shape} | Label: {label}")
print(f"{len(train_data.data)}, {len(train_data.targets)}, {len(test_data.data)}, {len(test_data.targets)}")
class_names = train_data.classes
print(f"Names: {class_names}")


# plt.imshow(image.squeeze(), cmap="gray")
# plt.title(label)
# plt.show()
fig = plt.figure(figsize=(9,9))
rows, cols = 4, 4
for i in range(1, rows * cols + 1):
    random_idx = torch.randint(0, len(train_data), size=[1]).item()
    img, label = train_data[random_idx]
    fig.add_subplot(rows, cols, i)
    plt.imshow(img.squeeze(), cmap="grey")
    plt.title(class_names[label])
    plt.axis(False)

plt.show()

# BATCH_SIZE = 32
# train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)
# test_dataloader = DataLoader(dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)

