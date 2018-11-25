from torchvision import datasets, transforms
from model.pixel_lstm.pixel_lstm import DiagonalPixelLSTM
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

BATCH_SIZE = 600
EPOCHS = 10
DEVICE = "cuda"

train_set = datasets.MNIST(
    "./MNIST",
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)
test_set = datasets.MNIST(
    "./MNIST",
    train=False,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    ),
)

train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)

model = nn.Sequential(
    nn.Conv2d(1, 10, [3, 3], padding=1),
    nn.ReLU(),
    DiagonalPixelLSTM(10, 50),
    nn.MaxPool2d([28, 28]),
    Flatten(),
    nn.Linear(50, 10),
)
model.to(DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.5)


def train_epoch(model, dataloader, optimizer):
    model.train()
    criterion = nn.CrossEntropyLoss()
    total_correct = 0
    total_loss = 0.0
    total_examples = 0
    for i, data in enumerate(dataloader):
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x)
        loss = criterion(y_hat, y)
        model.zero_grad()
        loss.backward()
        optimizer.step()

        total_examples += y.size(0)
        total_loss += loss.item()
        total_correct += (torch.argmax(y_hat, 1) == y).sum().item()

    return total_loss / len(dataloader), total_correct / total_examples


def test_epoch(model, dataloader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_correct = 0
    total_examples = 0
    total_loss = 0.0
    for i, data in enumerate(dataloader):
        x, y = data
        x, y = x.to(DEVICE), y.to(DEVICE)
        y_hat = model(x)
        loss = criterion(y_hat, y)

        total_loss += loss.item()
        total_correct += (torch.argmax(y_hat, 1) == y).sum().item()
        total_examples += y.size(0)

    return total_loss / len(dataloader), total_correct / total_examples


for epoch in range(EPOCHS):
    print("Epoch {:02d}".format(epoch))
    loss, acc = train_epoch(
        model, tqdm(train_loader, desc="Training", unit="batch"), optimizer
    )
    print("Train Loss: {:.04f} Accuracy: {:.04f}".format(loss, acc))
    loss, acc = test_epoch(model, test_loader)
    print("Test  Loss: {:.04f} Accuracy: {:.04f}".format(loss, acc))
print("Finished.")
