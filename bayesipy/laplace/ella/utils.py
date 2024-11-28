import numpy as np
import torch
from torch.utils.data import Sampler
import warnings

class SoDSampler(Sampler):
    def __init__(self, N, M, seed: int = 0):
        np.random.seed(seed)
        self.indices = torch.tensor(np.random.choice(list(range(N)), M, replace=False))

    def __iter__(self):
        return (i for i in self.indices)

    def __len__(self):
        return len(self.indices)


def psd_safe_eigen(K):
	Kprime = K.clone()
	jitter = 0
	jitter_new = None
	while True:
		p, q = torch.linalg.eigh(Kprime)
		if (p > 0).all():
			if jitter_new is not None:
				warnings.warn(
					f"K not p.d., added jitter of {jitter_new} to the diagonal",
					RuntimeWarning,
				)
			return p, q
		else:
			if jitter == 0:
				jitter_new = 1e-10
			else:
				jitter_new = jitter * 10
		Kprime.diagonal().add_(jitter_new - jitter)
		jitter = jitter_new

def subsample(loader, size, device):
    xs, ys = [], []
    i = 0
    for x_batch, y_batch in loader:
        for x, y in zip(x_batch, y_batch):
            if i >= size:
                break
            xs.append(x)
            ys.append(y)
            i += 1
    return torch.stack(xs).to(device), torch.stack(ys).to(device)

def subsample_balanced(loader, num_classes, subsample_number, device, verbose=False):
    print("Computing balanced subsample....")
    xs, ys = [], []
    cnt = np.zeros(num_classes)
    for x_batch, y_batch in loader:
        for x, y in zip(x_batch, y_batch):
            if np.all(cnt >= subsample_number//num_classes):
                xs = torch.stack(xs)
                ys = torch.stack(ys)
                if verbose:
                    print("The frequency of the sampled labels")
                    print(np.unique(ys.numpy(), return_counts=True))
                return xs.to(device), ys.to(device)
            if cnt[y.item()] >= subsample_number//num_classes:
                continue
            xs.append(x); ys.append(y)
            cnt[y.item()] += 1
            print(np.sum(cnt))

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
