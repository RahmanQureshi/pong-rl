import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):


    def __init__(self, num_input_channels):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(num_input_channels, 32, 8, stride=4),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU())
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Linear(16384, 512),
            nn.ReLU())
        self.fc2 = nn.Linear(512, 3)


    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(-1, self.num_flat_features(x)) # size => (1, 206*156)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


if __name__ == "__main__":
    print("Checking if forward function works")
    batch_size = 1
    x = torch.rand(batch_size, 3, 210, 160)
    net = Net()
    output = net(x)
    print("Output: {0}".format(output))
    print("Output size: {0}".format(output.size()))
    print("The size should be (1, 6)")