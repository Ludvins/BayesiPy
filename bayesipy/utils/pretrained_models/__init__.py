from .cnn import FMNIST_CNN, MNIST_CNN
from .mlp import Airline_MLP, Taxi_MLP, Year_MLP
from .resnet import CIFAR10_Resnet

__all__ = [
    "MNIST_CNN",
    "FMNIST_CNN",
    "CIFAR10_Resnet",
    "Airline_MLP",
    "Taxi_MLP",
    "Year_MLP",
]
