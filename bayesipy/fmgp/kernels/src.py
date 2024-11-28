import numpy as np
import torch


class SquaredExponential(torch.nn.Module):
    def __init__(
        self,
        initial_length_scale,
        initial_amplitude,
        n_features,
        n_outputs,
        device,
        dtype,
        embedding = None,
    ):
        super().__init__()
        self.device = device
        self.dtype = dtype
        self.n_features = n_features
        self.n_outputs = n_outputs
        
        self.embedding = embedding

        self.initial_amplitude = (
            torch.tensor(initial_amplitude).to(self.device).to(self.dtype)
        )
        self.initial_length_scale = (
            torch.tensor(initial_length_scale).to(self.device).to(self.dtype)
        )

        self.log_input_length_scale = torch.log(
            torch.tensor(initial_length_scale, device=self.device, dtype=self.dtype)
        ) * torch.ones(n_features, device=self.device, dtype=self.dtype)
        self.log_input_length_scale = torch.nn.Parameter(self.log_input_length_scale)

        self.log_input_amplitude = torch.log(
            torch.tensor(initial_amplitude, device=self.device, dtype=self.dtype)
        )
        self.log_input_amplitude = torch.nn.Parameter(self.log_input_amplitude)

        if n_outputs != 1:
            eye = np.exp(np.eye(self.n_outputs) - 1)

            # Initialize cholesky decomposition of identity
            li, lj = torch.tril_indices(self.n_outputs, self.n_outputs)
            # Shape (n_outputs, n_outputs)
            triangular_q_sqrt = eye[li, lj]
            # Shape (n_outputs, n_outputs)
            self.output_cholesky = torch.tensor(
                triangular_q_sqrt,
                dtype=self.dtype,
                device=self.device,
            )
            self.output_cholesky = torch.nn.Parameter(self.output_cholesky)

    def __call__(self, x1, x2=None, diag=False):
        """Computes the RBF kernel

        Parameters
        ----------

        x1 : Torch tensor of shape (n, input_dim)
            Contains the features of the first input
        x2 : Torch tensor of shape (m, input_dim)
            Contains the features of the second input

        """
        if x2 is None:
            x2 = x1
            
        if self.embedding is not None:
            x1 = self.embedding(x1)
            x2 = self.embedding(x2)

        # Scale the input features using the length scale
        x1 = x1 / torch.exp(self.log_input_length_scale)
        x2 = x2 / torch.exp(self.log_input_length_scale)

        B1 = x1.shape[0]
        B2 = x2.shape[0]

        # Make x1 shape (n, 1, input_dim) and x2 shape (1, m, input_dim)
        #  the diff has shape (n, m, input_dim) due to broadcast
        #  Take squared and add on last dimension.
        # dist = torch.sum((x1.unsqueeze(1) - x2.unsqueeze(0))**2, -1)
        if diag:
            if x1.equal(x2):
                dist = torch.zeros(B1, device=self.device, dtype=self.dtype)
            else:
                dist = (x1.reshape(B1, -1) - x2.reshape(B2, -1)) ** 2
                dist = torch.sum(dist, -1)
        else:
            dist = torch.cdist(x1.reshape(B1, -1), x2.reshape(B2, -1)) ** 2
        kernel = torch.exp(self.log_input_amplitude) * torch.exp(-dist / 2)

        if self.n_outputs != 1:
            outputs_kernel = self.compute_output_kernel()
            return kernel.unsqueeze(-1).unsqueeze(-1) * outputs_kernel

        return kernel

    def compute_output_kernel(self):
        if self.n_outputs == 1:
            raise ValueError("Kernel with a single output dimension has no output fun")

        L = torch.eye(self.n_outputs, dtype=self.dtype, device=self.device)
        li, lj = torch.tril_indices(self.n_outputs, self.n_outputs)
        # Shape (n_outputs, n_outputs)
        L[li, lj] = self.output_cholesky

        return L @ L.T


class LastLayerNTK_SquaredExponential(SquaredExponential):
    def __init__(
        self,
        initial_length_scale,
        initial_amplitude,
        n_features,
        n_outputs,
        device,
        dtype,
    ):
        super().__init__(
            initial_length_scale,
            initial_amplitude,
            n_features,
            n_outputs,
            device,
            dtype,
        )
        # self.log_ntk_amplitude = torch.log(
        #     torch.tensor(initial_amplitude, device=device, dtype=dtype)
        # )
        # self.log_ntk_amplitude = torch.nn.Parameter(self.log_ntk_amplitude)

    def NTK(self, f1, f2=None, diag=False):
        if f2 is None:
            f2 = f1

        # Repeat f1 n_outputs times. Shape (n, n_features, n_outputs)
        J1 = f1.unsqueeze(-1).expand(-1, -1, self.n_outputs)

        # Concatenate an indentity matrix to J1 of shape (n, n_features + n_outputs, n_outputs)
        eye = torch.eye(self.n_outputs, dtype=self.dtype, device=self.device)
        eye = eye.unsqueeze(0).expand(J1.shape[0], -1, -1)
        J1 = torch.cat([J1, eye], dim=1)

        # Repeat f2 n_outputs times. Shape (m, n_features, n_outputs)
        J2 = f2.unsqueeze(-1).expand(-1, -1, self.n_outputs)
        # Concatenate an indentity matrix to J2 of shape (m, n_features + n_outputs, n_outputs)
        eye = torch.eye(self.n_outputs, dtype=self.dtype, device=self.device)
        eye = eye.unsqueeze(0).expand(J2.shape[0], -1, -1)
        J2 = torch.cat([J2, eye], dim=1)

        if diag:
            K = torch.einsum("apo, apu -> aou", J1, J2)
        else:
            K = torch.einsum("apo, bpu -> abou", J1, J2)
        return K  # * torch.exp(self.log_ntk_amplitude)

    def __call__(self, xf1, xf2=None, diag=False):
        """Computes the RBF kernel

        Parameters
        ----------

        x1 : Torch tensor of shape (n, input_dim)
            Contains the features of the first input
        x2 : Torch tensor of shape (m, input_dim)
            Contains the features of the second input

        """
        x1, f1 = xf1
        x2, f2 = xf2 or xf1
        K = super().__call__(x1, x2, diag)
        K2 = self.NTK(f1, f2, diag)
        return K * K2
