
from torch import nn
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.channels1 = 2
        self.channels2 = 2

        self.conv1 = nn.Conv2d(1, self.channels1, 5, 1)
        self.conv2 = nn.Conv2d(self.channels1, self.channels2, 5, 1)
        self.fc1 = nn.Linear(4*4*self.channels2, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4*4*self.channels2)
        x = self.fc1(x)
        return x


class Linear(nn.Module):
    def __init__(self):
        super(Linear, self).__init__()

        self.fc1 = nn.Linear(784, 400)
        self.fc2 = nn.Linear(400, 100)
        self.fc3 = nn.Linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
