"""
refer to: https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as dset
import torchvision.transforms as transforms


def load_data(batch_size, worker_index):
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # We don't use the following function because we want to 
    # devide the training set by ourselves.
    # trainset = torchvision.datasets.CIFAR10(root='/tmp', train=True,
    #                                         download=True, transform=transform)

    # use ImageFolder to turn pictures into binary files.
    # path is created using "divide.py".
    pathname = '/home/starfish/workspace/Distributed_DOCKER/file/data/cifar-10-pictures-' + \
        str(worker_index)
    trainset = dset.ImageFolder(pathname, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                              shuffle=True, num_workers=0)

    # testset = torchvision.datasets.CIFAR10(root='/tmp', train=False,
    #                                        download=True, transform=transform)
    testset = dset.ImageFolder(
        '/home/starfish/workspace/Distributed_DOCKER/file/data/cifar-10-pictures-test', transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                             shuffle=False, num_workers=0)

    return trainloader, testloader


# classes = ('plane', 'car', 'bird', 'cat',
#            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# The following is the neural network we use.
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
