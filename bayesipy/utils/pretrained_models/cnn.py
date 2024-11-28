import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from bayesipy import ROOT_DIR


class CNN(nn.Module):
    def __init__(self, input_channels):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        return self.fc2(x)


def MNIST_CNN(embedding=False, classifier=False, get_transform=False):
    if get_transform:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

    if embedding and classifier:
        raise ValueError("Only one of embedding and classifier can be True")

    net = CNN(input_channels=1)
    net.load_state_dict(
        torch.load(ROOT_DIR + "utils/pretrained_models/weights/mnist_cnn.pt")
    )

    if embedding:
        net.fc2 = nn.Identity()
    elif classifier:
        net = net.fc2
    net.eval()
    return net


def FMNIST_CNN(embedding=False, classifier=False, get_transform=False):
    if get_transform:
        return torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize((0.5,), (0.5,)),
            ]
        )

    if embedding and classifier:
        raise ValueError("Only one of embedding and classifier can be True")

    net = CNN(input_channels=1)
    net.load_state_dict(
        torch.load(ROOT_DIR + "utils/pretrained_models/weights/fmnist_cnn.pt")
    )

    if embedding:
        net.fc2 = nn.Identity()
    elif classifier:
        net = net.fc2
    net.eval()
    return net
