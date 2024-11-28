import torch

from bayesipy import ROOT_DIR

root = ROOT_DIR + "utils/pretrained_models/weights/"


def get_mlp(
    input_dim,
    output_dim,
    inner_dims,
    activation,
):
    torch.manual_seed(2147483647)

    layers = []
    dims = [input_dim] + inner_dims + [output_dim]
    for i, (_in, _out) in enumerate(zip(dims[:-1], dims[1:])):
        layers.append(torch.nn.Linear(_in, _out))

        if i != len(dims) - 2:
            layers.append(activation())

    model = torch.nn.Sequential(*layers)
    return model


def Airline_MLP():
    model = get_mlp(
        input_dim=8,
        output_dim=1,
        inner_dims=[200, 200, 200],
        activation=torch.nn.Tanh,
    )
    model.load_state_dict(torch.load(root + "airline_mlp.pt"))
    return model


def Taxi_MLP():
    model = get_mlp(
        input_dim=8,
        output_dim=1,
        inner_dims=[200, 200, 200],
        activation=torch.nn.Tanh,
    )
    model.load_state_dict(torch.load(root + "taxi_mlp.pt"))
    return model


def Year_MLP():
    model = get_mlp(
        input_dim=90,
        output_dim=1,
        inner_dims=[200, 200, 200],
        activation=torch.nn.Tanh,
    )
    model.load_state_dict(torch.load(root + "year_mlp.pt"))
    return model
