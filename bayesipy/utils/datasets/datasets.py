import os
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
import torch.utils
import torchvision
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import datasets, transforms

from bayesipy import ROOT_DIR

uci_base = "https://archive.ics.uci.edu/ml/machine-learning-databases/"
data_base = ROOT_DIR + "utils/datasets/data/"


class Training_Dataset(Dataset):
    def __init__(
        self,
        inputs,
        targets,
        output_dim,
        verbose=True,
        normalize_inputs=True,
        normalize_targets=True,
    ):
        self.inputs = inputs
        if normalize_targets:
            self.targets_mean = np.mean(targets, axis=0, keepdims=True)
            self.targets_std = np.std(targets, axis=0, keepdims=True)
        else:
            self.targets_mean = 0
            self.targets_std = 1
        self.targets = (targets - self.targets_mean) / self.targets_std

        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1:]
        self.output_dim = output_dim

        # Normalize inputs
        if normalize_inputs:
            self.inputs_std = np.std(self.inputs, axis=0, keepdims=True) + 1e-6
            self.inputs_mean = np.mean(self.inputs, axis=0, keepdims=True)
        else:
            self.inputs_mean = 0
            self.inputs_std = 1

        self.inputs = (self.inputs - self.inputs_mean) / self.inputs_std
        if verbose:
            print("Number of samples: ", self.n_samples)
            print("Input dimension: ", self.input_dim)
            print("Label dimension: ", self.output_dim)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class Input_Output_Dataset(Dataset):
    def __init__(
        self,
        inputs,
        outputs,
        targets,
        output_dim,
    ):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)

        self.targets = np.array(targets)

        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        self.output_dim = output_dim

    def __getitem__(self, index):
        return (self.inputs[index], self.outputs[index]), self.targets[index]

    def __len__(self):
        return len(self.inputs)


class Input_Embedding_Output_Dataset(Dataset):
    def __init__(
        self,
        inputs,
        embedding,
        outputs,
        targets,
        output_dim,
    ):
        self.inputs = np.array(inputs)
        self.outputs = np.array(outputs)
        self.embedding = np.array(embedding)

        self.targets = np.array(targets)

        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1:]
        self.embedding_dim = embedding.shape[1:]
        self.output_dim = output_dim

    def __getitem__(self, index):
        return (
            (self.inputs[index], self.embedding[index], self.outputs[index]),
            self.targets[index],
        )

    def __len__(self):
        return len(self.inputs)


class Test_Dataset(Dataset):
    def __init__(
        self, inputs, output_dim, targets=None, inputs_mean=0.0, inputs_std=1.0
    ):
        self.inputs = (inputs - inputs_mean) / inputs_std
        self.targets = targets
        self.n_samples = inputs.shape[0]
        self.input_dim = inputs.shape[1]
        if self.targets is not None:
            self.output_dim = output_dim

    def __getitem__(self, index):
        if self.targets is None:
            return self.inputs[index]
        return self.inputs[index], self.targets[index]

    def __len__(self):
        return len(self.inputs)


class SPGP_Dataset:
    def __init__(self):
        self.output_dim = 1
        inputs = np.loadtxt(data_base + "SPGP_dist/train_inputs")
        targets = np.loadtxt(data_base + "SPGP_dist/train_outputs")

        mask = ((inputs < 1.5) | (inputs > 3.5)).flatten()
        mask2 = ((inputs >= 1.5) & (inputs <= 3.5)).flatten()

        self.train = Training_Dataset(
            inputs[mask, np.newaxis],
            targets[mask, np.newaxis],
            normalize_targets=False,
            normalize_inputs=True,
        )

        self.test = Test_Dataset(
            inputs[mask2, np.newaxis],
            targets[mask2, np.newaxis],
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def get_splits(self):
        return self.train, self.test

    def train_size(self):
        return 200


class Synthetic_Dataset:
    def __init__(self):
        self.output_dim = 1
        data = np.load(data_base + "synthetic_data.npy")

        inputs, targets = data[:, 0], data[:, 1]

        test_inputs = np.linspace(np.min(inputs) - 5, np.max(inputs) + 5, 200)

        self.train = Training_Dataset(
            inputs[..., np.newaxis],
            targets[..., np.newaxis],
            output_dim=1,
            normalize_targets=False,
            normalize_inputs=False,
        )

        self.test = Test_Dataset(
            test_inputs[..., np.newaxis],
            1,
            None,
            self.train.inputs_mean,
            self.train.inputs_std,
        )

    def get_splits(self):
        return self.train, self.test

    def train_size(self):
        return 400


class MNIST_Dataset:
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        self.classes = 10
        self.output_dim = 10
        self.rotation_angles = np.arange(10, 181, 10)

        self.train = datasets.MNIST(
            root=data_base, train=True, download=True, transform=transform
        )
        self.test = datasets.MNIST(
            root=data_base,
            train=False,
            download=True,
            transform=transform,
        )

    def get_splits(self):
        return self.train, self.test

    def train_size(self):
        return len(self.train)


class MNIST_OOD_Dataset:
    def __init__(self, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        self.classes = 1
        self.output_dim = 1

        self.train = datasets.MNIST(
            root=data_base, train=True, download=True, transform=transform
        )
        test = datasets.MNIST(
            root=data_base,
            train=False,
            download=True,
            transform=transform,
        )
        fmnist_test = datasets.FashionMNIST(
            root=data_base,
            train=False,
            download=True,
            transform=transform,
        )

        test_loader = DataLoader(test, batch_size=len(test))
        test_inputs, _ = next(iter(test_loader))
        test2_loader = DataLoader(fmnist_test, batch_size=len(test))
        test2_inputs, _ = next(iter(test2_loader))

        ood_test_data = np.concatenate([test_inputs, test2_inputs])
        ood_test_targets = np.concatenate(
            [np.zeros(test_inputs.shape[0]), np.ones(test2_inputs.shape[0])]
        ).reshape(-1, 1)

        self.test = TensorDataset(ood_test_data, ood_test_targets)

    def get_splits(self):
        return self.train, self.test

    def train_size(self):
        return len(self.train)


class MNIST_Rotated_Dataset:
    def __init__(self, angle, transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        # Concatenate rotation to transform
        transform = transforms.Compose(
            [transform, transforms.RandomRotation(angle, angle)]
        )

        self.classes = 1
        self.output_dim = 1

        self.train = datasets.MNIST(
            root=data_base, train=True, download=True, transform=transform
        )
        self.test = datasets.MNIST(
            root=data_base,
            train=False,
            download=True,
            transform=transform,
        )

    def get_split(self):
        return self.train, self.test

    def len_train(self):
        return len(self.train)


class CIFAR10_Dataset:
    def __init__(self, data_dir="./data/", transform=None):
        self.classes = 10
        self.output_dim = 10
        self.data_dir = data_dir

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.train = datasets.CIFAR10(
            root=data_dir, train=True, download=True, transform=transform
        )
        self.test = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )

    def train_test_splits(self):
        return self.train, self.test

    def validation_split(self, lower=0.5, size=5000):
        assert size <= len(
            self.train
        ), f"Size exceeds training set size: {self.len_train()}"
        transform = self.train.transform

        transform = transforms.Compose(
            [transforms.RandomResizedCrop(size=32, scale=(lower, 1)), transform]
        )

        valid_dataset = datasets.CIFAR10(
            root=self.data_dir, train=True, transform=transform, download=True
        )

        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=size, shuffle=False
        )

        X, y = next(iter(val_loader))
        return torch.utils.data.TensorDataset(X, y)

    def len_train(self):
        return len(self.train)


class CIFAR10_Rotated_Dataset:
    def __init__(self, angle, data_dir="./data/", transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        # Concatenate rotation to transform
        transform = transforms.Compose(
            [transform, transforms.RandomRotation(degrees=angle)]
        )

        self.train = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        self.test = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )

    def train_test_splits(self):
        return self.train, self.test

    def len_train(self):
        return len(self.train)


class CIFAR10_OOD_Dataset:
    def __init__(self,  data_dir="./data/", transform=None):
        if transform is None:
            transform = transforms.ToTensor()

        self.train = datasets.CIFAR10(
            root=data_dir,
            train=True,
            download=True,
            transform=transform,
        )
        
        test = datasets.CIFAR10(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )
        test2 = datasets.SVHN(
            root=data_dir,
            train=False,
            download=True,
            transform=transform,
        )

        test_loader = DataLoader(test, batch_size=len(test))
        test_inputs, _ = next(iter(test_loader))
        test2_loader = DataLoader(test2, batch_size=len(test))
        test2_inputs, _ = next(iter(test2_loader))

        ood_test_data = np.concatenate([test_inputs, test2_inputs])
        ood_test_targets = np.concatenate(
            [np.zeros(test_inputs.shape[0]), np.ones(test2_inputs.shape[0])]
        ).reshape(-1, 1)

        self.test = TensorDataset(ood_test_data, ood_test_targets)

    def get_splits(self):
        return self.train, self.test

    def train_size(self):
        return len(self.train)


class Precomputed_Output_Dataset:
    def __init__(self, model, model_name, dataset, data_dir="./data/", transform=None):
        self.classes = 10
        self.output_dim = 10

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        train, test = dataset(
            transform=transform, data_dir=data_dir
        ).train_test_splits()

        # Check if file exists
        if not os.path.isfile(data_dir + f"cifar10_{model_name}.npy"):
            print("Computing predictions for CIFAR10")
            preds = []
            for i in range(0, len(train.inputs), 100):
                input = torch.tensor(train.inputs[i : i + 100]).float()
                preds.append(model(input).detach().numpy())
            preds = np.concatenate(preds)
            np.save(f"./data/cifar10_{model_name}.npy", preds)
        else:
            print("Loading precomputed predictions for CIFAR10")
            preds = np.load(data_dir + f"cifar10_{model_name}.npy")

        # Check if file exists
        if not os.path.isfile(data_dir + f"cifar10_test_{model_name}.npy"):
            print("Computing predictions for CIFAR10 Test")
            preds_test = []
            for i in range(0, len(test.inputs), 100):
                preds_test.append(
                    model(torch.tensor(test.inputs[i : i + 100]).float())
                    .detach()
                    .numpy()
                )
            preds_test = np.concatenate(preds_test)
            np.save(data_dir + f"cifar10_test_{model_name}.npy", preds_test)
        else:
            print("Loading precomputed predictions for CIFAR10")
            preds_test = np.load(data_dir + f"cifar10_test_{model_name}.npy")

        self.train = Input_Output_Dataset(
            train.data,
            preds,
            train.targets,
            self.output_dim,
        )
        self.test = Input_Output_Dataset(
            test.data,
            preds_test,
            test.targets,
            self.output_dim,
        )

    def get_split(self):
        return self.train, self.test

    def len_train(self):
        return len(self.train)


class Precomputed_Output_Embedding_Dataset:
    def __init__(
        self, model, embedding, model_name, dataset, data_dir="./data/", transform=None
    ):
        self.classes = 10
        self.output_dim = 10

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        train, test = dataset(
            transform=transform, data_dir=data_dir
        ).train_test_splits()
        # Get dtype of a parameter
        DTYPE = next(model.parameters()).dtype
        DEVICE = next(model.parameters()).device
        # Check if file exists
        if not os.path.isfile(data_dir + f"cifar10_{model_name}_embedding.npy"):
            print("Computing embedding for CIFAR10")
            embedding_train = []
            loader = DataLoader(train, batch_size=100, shuffle=False)
            for inputs, _ in loader:
                embedding_train.append(
                    embedding(inputs.to(DTYPE).to(DEVICE))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(100, -1)
                )
            embedding_train = np.concatenate(embedding_train)
            np.save(data_dir + f"cifar10_{model_name}_embedding.npy", embedding_train)
        else:
            print("Loading precomputed predictions for CIFAR10")
            embedding_train = np.load(data_dir + f"cifar10_{model_name}_embedding.npy")

        # Check if file exists
        if not os.path.isfile(data_dir + f"cifar10_test_{model_name}_embedding.npy"):
            print("Computing embedding for CIFAR10 Test")
            embedding_test = []
            loader = DataLoader(test, batch_size=100, shuffle=False)
            for inputs, _ in loader:
                embedding_test.append(
                    embedding(inputs.to(DTYPE).to(DEVICE))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(100, -1)
                )
            embedding_test = np.concatenate(embedding_test)
            np.save(
                data_dir + f"cifar10_test_{model_name}_embedding.npy", embedding_test
            )
        else:
            print("Loading precomputed predictions for CIFAR10")
            embedding_test = np.load(
                data_dir + f"cifar10_test_{model_name}_embedding.npy"
            )

        # Check if file exists
        if not os.path.isfile(data_dir + f"cifar10_{model_name}.npy"):
            print("Computing predictions for CIFAR10")
            preds = []
            loader = DataLoader(train, batch_size=100, shuffle=False)
            for inputs, _ in loader:
                preds.append(
                    model(inputs.to(DTYPE).to(DEVICE))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(100, -1)
                )
            preds = np.concatenate(preds)
            np.save(data_dir + f"cifar10_{model_name}.npy", preds)
        else:
            print("Loading precomputed predictions for CIFAR10")
            preds = np.load(data_dir + f"cifar10_{model_name}.npy")

        # Check if file exists
        if not os.path.isfile(data_dir + f"cifar10_test_{model_name}.npy"):
            print("Computing predictions for CIFAR10 Test")
            preds_test = []
            loader = DataLoader(test, batch_size=100, shuffle=False)
            for inputs, _ in loader:
                preds_test.append(
                    model(inputs.to(DTYPE).to(DEVICE))
                    .detach()
                    .cpu()
                    .numpy()
                    .reshape(100, -1)
                )
            preds_test = np.concatenate(preds_test)
            np.save(data_dir + f"cifar10_test_{model_name}.npy", preds_test)
        else:
            print("Loading precomputed predictions for CIFAR10")
            preds_test = np.load(data_dir + f"cifar10_test_{model_name}.npy")

        train_data = []
        test_data = []
        loader = DataLoader(train, batch_size=100, shuffle=False)
        for inputs, _ in loader:
            train_data.append(inputs.numpy())
        loader = DataLoader(test, batch_size=100, shuffle=False)
        for inputs, _ in loader:
            test_data.append(inputs.numpy())
        train_data = np.concatenate(train_data)
        test_data = np.concatenate(test_data)

        self.train = Input_Embedding_Output_Dataset(
            train_data,
            embedding_train,
            preds,
            train.targets,
            self.output_dim,
        )
        self.test = Input_Embedding_Output_Dataset(
            test_data,
            embedding_test,
            preds_test,
            test.targets,
            self.output_dim,
        )

    def train_test_splits(self):
        return self.train, self.test

    def len_train(self):
        return len(self.train)


class Airline_Dataset:
    def __init__(self):
        self.output_dim = 1

        data1 = pd.read_csv(data_base + "airline_1.csv")
        data2 = pd.read_csv(data_base + "airline_2.csv")
        data = pd.concat([data1, data2])
        # data = pd.read_csv(data_base + "airline.csv")
        # print(data)
        # print(data2)
        # print(data.equals(data2))
        # Convert time of day from hhmm to minutes since midnight
        data.ArrTime = 60 * np.floor(data.ArrTime / 100) + np.mod(data.ArrTime, 100)
        data.DepTime = 60 * np.floor(data.DepTime / 100) + np.mod(data.DepTime, 100)

        # Pick out the data
        Y = data["ArrDelay"].values[:800000].reshape(-1, 1)
        names = [
            "Month",
            "DayofMonth",
            "DayOfWeek",
            "plane_age",
            "AirTime",
            "Distance",
            "ArrTime",
            "DepTime",
        ]
        X = data[names].values[:800000]
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)

        self.n_train = 600000
        self.n_val = 100000

        self.X_mean = np.mean(X[: self.n_train], axis=0, keepdims=True)
        self.X_std = np.std(X[: self.n_train], axis=0, keepdims=True)

        self.y_mean = np.mean(Y[: self.n_train], axis=0, keepdims=True)
        self.y_std = np.std(Y[: self.n_train], axis=0, keepdims=True)

        X = (X - self.X_mean) / self.X_std
        Y[: self.n_train] = (Y[: self.n_train] - self.y_mean) / self.y_std

        self.train = Training_Dataset(
            X[: self.n_train], Y[: self.n_train], self.output_dim
        )
        self.input_dim = X.shape[1]

        self.val = Test_Dataset(
            X[self.n_train : self.n_train + self.n_val],
            self.output_dim,
            Y[self.n_train : self.n_train + self.n_val],
        )
        self.test = Test_Dataset(
            X[self.n_train + self.n_val :],
            self.output_dim,
            Y[self.n_train + self.n_val :],
        )

    def train_size(self):
        return self.n_train

    def train_test_splits(self):
        return self.train, self.test

    def validation_split(self):
        return self.val


class Year_Dataset:
    def __init__(self, data_dir="./data/"):
        self.output_dim = 1
        if not os.path.exists(data_dir + "YearPredictionMSD.txt"):
            url = "{}{}".format(uci_base, "00203/YearPredictionMSD.txt.zip")
            with urlopen(url) as zipresp:
                with ZipFile(BytesIO(zipresp.read())) as zfile:
                    zfile.extractall(data_dir)

        data = pd.read_csv(
            data_dir + "YearPredictionMSD.txt", header=None, delimiter=","
        ).values

        self.len_data = data.shape[0]

        X = data[:, 1:]

        self.input_dim = X.shape[1]
        Y = data[:, 0].reshape(-1, 1)
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)

        self.n_train = 400000
        self.n_val = 63715

        self.X_mean = np.mean(X[: self.n_train], axis=0, keepdims=True) + 1e-6
        self.X_std = np.std(X[: self.n_train], axis=0, keepdims=True)

        self.y_mean = np.mean(Y[: self.n_train], axis=0, keepdims=True)
        self.y_std = np.std(Y[: self.n_train], axis=0, keepdims=True)

        X = (X - self.X_mean) / self.X_std
        Y[: self.n_train] = (Y[: self.n_train] - self.y_mean) / self.y_std

        self.train = TensorDataset(
            torch.tensor(X[: self.n_train]),
            torch.tensor(Y[: self.n_train]),
        )
        self.val = TensorDataset(
            torch.tensor(X[self.n_train : self.n_train + self.n_val]),
            torch.tensor(Y[self.n_train : self.n_train + self.n_val]),
        )
        self.test = TensorDataset(
            torch.tensor(X[self.n_train + self.n_val :]),
            torch.tensor(Y[self.n_train + self.n_val :]),
        )

    def train_size(self):
        return self.n_train

    def train_test_splits(self):
        return self.train, self.test

    def validation_split(self):
        return self.val


class Taxi_Dataset:
    def __init__(self, data_dir="./data/"):
        self.output_dim = 1

        if os.path.exists(data_dir + "taxi.csv"):
            print("Taxi csv file found.")
            data = pd.read_csv(data_dir + "taxi.csv")
        elif os.path.exists(data_dir + "taxi.zip"):
            print("Taxi zip file found.")
            data = pd.read_csv(data_dir + "taxi.zip", compression="zip", dtype=object)
        else:
            print("Downloading Taxi Dataset...")
            url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
            data = pd.read_parquet(url)
            data.to_csv(data_dir + "taxi.csv")

        data["tpep_pickup_datetime"] = pd.to_datetime(data["tpep_pickup_datetime"])
        data["tpep_dropoff_datetime"] = pd.to_datetime(data["tpep_dropoff_datetime"])
        data["day_of_week"] = data["tpep_pickup_datetime"].dt.dayofweek
        data["day_of_month"] = data["tpep_pickup_datetime"].dt.day
        data["month"] = data["tpep_pickup_datetime"].dt.month

        data["time_of_day"] = (
            data["tpep_pickup_datetime"] - data["tpep_pickup_datetime"].dt.normalize()
        ) / pd.Timedelta(seconds=1)
        data["trip_duration"] = (
            data["tpep_dropoff_datetime"] - data["tpep_pickup_datetime"]
        ).dt.total_seconds()
        data = data[
            [
                "time_of_day",
                "day_of_week",
                "day_of_month",
                "month",
                "PULocationID",
                "DOLocationID",
                "trip_distance",
                "trip_duration",
                "total_amount",
            ]
        ]
        data = data[data["trip_duration"] >= 10]
        data = data[data["trip_duration"] <= 5 * 3600]
        data = data.astype(float)
        data = data.values
        X = data[:, :-1]
        Y = data[:, -1][:, np.newaxis]
        X = X.astype(np.float64)
        Y = Y.astype(np.float64)
        self.input_dim = X.shape[1]

        self.n_train = int(X.shape[0] * 0.8)
        self.n_val = int(X.shape[0] * 0.1)

        self.X_mean = np.mean(X[: self.n_train], axis=0, keepdims=True) + 1e-6
        self.X_std = np.std(X[: self.n_train], axis=0, keepdims=True)

        self.y_mean = np.mean(Y[: self.n_train], axis=0, keepdims=True)
        self.y_std = np.std(Y[: self.n_train], axis=0, keepdims=True)

        X = (X - self.X_mean) / self.X_std
        Y[: self.n_train] = (Y[: self.n_train] - self.y_mean) / self.y_std

        self.train = TensorDataset(
            torch.tensor(X[: self.n_train]),
            torch.tensor(Y[: self.n_train]),
        )
        self.val = TensorDataset(
            torch.tensor(X[self.n_train : self.n_train + self.n_val]),
            torch.tensor(Y[self.n_train : self.n_train + self.n_val]),
        )
        self.test = TensorDataset(
            torch.tensor(X[self.n_train + self.n_val :]),
            torch.tensor(Y[self.n_train + self.n_val :]),
        )

    def train_size(self):
        return self.n_train

    def train_test_splits(self):
        return self.train, self.test

    def validation_split(self):
        return self.val


class Imagenet_Dataset:
    def __init__(self, data_dir="./data/", transform=None):
        self.data_dir = data_dir

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        self.train = torchvision.datasets.ImageNet(
            root=data_dir, split="train", transform=transform
        )
        self.test = torchvision.datasets.ImageNet(
            root=data_dir, split="val", transform=transform
        )

    def train_test_splits(self):
        return self.train, self.test

    def validation_split(self, lower=0.5, size=5000):
        assert size <= len(
            self.train
        ), f"Size exceeds training set size: {self.len_train()}"
        transform = self.train.transform

        da = transforms.Compose(
            [
                torchvision.transforms.PILToTensor(),
                transforms.RandomResizedCrop(size=224, scale=(lower, 1)),
            ]
        )

        valid_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir, split="train", transform=da
        )

        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=size, shuffle=False
        )

        X, y = next(iter(val_loader))
        X = transform(X)
        return torch.utils.data.TensorDataset(X, y)

    def validation_split_ella(self, lower=0.5, size=12000):
        assert size <= len(
            self.train
        ), f"Size exceeds training set size: {self.len_train()}"
        transform = create_transform(
            input_size=224,
            scale=(0.08, 0.1),
            is_training=True,
            color_jitter=0.4,
            auto_augment=None,
            interpolation="bicubic",
            re_prob=0.25,
            re_mode="pixel",
            re_count=1,
            mean=IMAGENET_DEFAULT_MEAN,
            std=IMAGENET_DEFAULT_STD,
        )

        valid_dataset = torchvision.datasets.ImageNet(
            root=self.data_dir, split="train", transform=transform
        )

        val_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=size, shuffle=False
        )

        X, y = next(iter(val_loader))
        return torch.utils.data.TensorDataset(X, y)

    def len_train(self):
        return len(self.train)


class Rotated_Imagenet_Dataset:
    def __init__(self, angle, data_dir="./data/", transform=None):
        self.classes = 1000
        self.output_dim = 1000

        if transform is None:
            transform = transforms.Compose([transforms.ToTensor()])

        transform = transforms.Compose(
            [transform, transforms.RandomRotation(degrees=angle)]
        )

        self.train = torchvision.datasets.ImageNet(
            root=data_dir, split="train", transform=transform
        )
        self.test = torchvision.datasets.ImageNet(
            root=data_dir, split="val", transform=transform
        )

    def train_test_splits(self):
        return self.train, self.test

    def len_train(self):
        return len(self.train)
