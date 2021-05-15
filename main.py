import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
from torchvision.transforms import ToTensor, Lambda, Compose
import torch.optim as optim

# Download data if not available locally and create a dataset
training_data = datasets.FashionMNIST(
    root='data',
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download data if not available locally and create a dataset
test_data = datasets.FashionMNIST(
    root='data',
    train=False,
    download=True,
    transform=ToTensor(),
)

batch_size = 64

# create data loaders
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

# Check the shapes of features and targets
for X, y in test_dataloader:
    print('Shape of X [N, C, H, W]: ', X.shape)
    print('Shape of y: ', y.shape)
    break

# set device to cuda if available
device = "cpu"
if torch.cuda.is_available():
    device = "cuda"
print(f'Using {device} device...')


# Define model
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Instantiate model
model = Network().to(device)
print(model)

# optimizer and loss function
# To train a model we need an optimizer and a loss function
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)


# Training loop - runs once for each epoch
def train(dataloader, model, loss_fn, optimizer):
    sz = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch%100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f'loss: {loss:>7f} [{current:>5d}/{sz:>5d}]')


# Evaluate model's performance on test data
def test(dataloader, model):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        test_loss /= size
        correct /= size
        print(f'Test stats: Accuracy: {(100*correct):>0.1f}, Avg loss: {test_loss:>8f}')


epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model)
print('Done')


# Saving the model
# A common way to save a model is to serialize the internal state dictionary (containing the model parameters)
torch.save(model.state_dict(), "model.pth")
print(f'saved the model to model.pth')

# Loading the model
# The process for loading a model includes recreating the model structure and loading the state dictionary into it.
model = Network()
model.load_state_dict(torch.load("model.pth"))

# Prediction
classes = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot",
]

model.eval()
x, y = test_data[0][0], test_data[0][1]
with torch.no_grad():
    pred = model(x)
    predicted, actual = classes[pred[0].argmax(0)], classes[y]
    print(f'Predicted: {predicted}, Actual: {actual}')
