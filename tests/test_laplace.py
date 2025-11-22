
import torch

from bayesipy.laplace import Laplace


def test_laplace_pipeline() -> None:
    # Tiny model and data
    model = torch.nn.Sequential(
        torch.nn.Linear(4, 16),
        torch.nn.ReLU(),
        torch.nn.Linear(16, 3),
    )
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))

    laplace = Laplace(model, "classification", subset_of_weights="all", hessian_structure="full")
    train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x, y), batch_size=4)
    laplace.fit(train_loader)
    fmean, fvar = laplace.predict(x)  

    assert fmean.shape == (8, 3)
    assert fvar.shape == (8, 3, 3)
