import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 1, 5) 
        self.fc1 = nn.Linear(206 * 156, 6) 

    def forward(self, x):
        # x.size() => (210, 156)
        x = F.relu(self.conv1(x)) # size => (206, 156)
        x = x.view(-1, self.num_flat_features(x)) # size => (1, 206*156)
        x = self.fc1(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

def train(net, X, Y, num_epochs):
    optimizer = optim.SGD(net.parameters(), lr=0.01)
    for i in range(0, num_epochs):
        optimizer.zero_grad()
        output = net(X)
        loss = criterion(output, Y)
        loss.backward()
        optimizer.step()    # Does the update

class PongAgent:
    """
    Observation space: Box(210, 160, 3)
    Action space: Discrete(6)
    """

    def __init__(self, net):
        self.net = net

    def getAction(self, observation):
        output = self.net(observation)
        return output.argmax().item()


if __name__ == "__main__":
    print("Checking if forward function works")
    batch_size = 1
    x = torch.rand(batch_size, 3, 210, 160)
    net = Net()
    output = net(x)
    print("Output: {0}".format(output))
    print("Output size: {0}".format(output.size()))
    print("The size should be (1, 6)")