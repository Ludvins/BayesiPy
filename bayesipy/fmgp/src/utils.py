import torch


def compute_length_scale_estimation(loader, n_samples=50000):
    distances = []
    data_iter = iter(loader)

    B = loader.batch_size

    for i in range(n_samples // B):
        try:
            inputs = next(data_iter)[0]
            inputs2 = next(data_iter)[0]
        except StopIteration:
            # StopIteration is thrown if dataset ends
            # reinitialize data loader
            data_iter = iter(loader)
            inputs = next(data_iter)[0]
            inputs2 = next(data_iter)[0]

        if B == 1:
            inputs = inputs.unsqueeze(0)
            inputs2 = inputs2.unsqueeze(0)
        for a, b in zip(inputs, inputs2):
            # Compute distance using numpy
            dist = torch.linalg.vector_norm(a.flatten() - b.flatten(), ord=2) ** 2
            distances.append(dist)
    distances = torch.stack(distances)
    return torch.sqrt(torch.median(distances) / 2).item()
