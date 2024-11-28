import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .utils import MulExpAddFunction


class _BayesBatchNormMF(Module):
    """
    Applies Bayesian Batch Normalization over a 2D or 3D input

    Parameters
    ----------
    num_features : int
        Number of features in the input.
    eps : float, optional
        Small value added to the denominator for numerical stability. Default: 1e-5.
    momentum : float, optional
        Value used for the running mean and running variance computation. Default: 0.1.
    affine : bool, optional
        If True, this module has learnable affine parameters. Default: True.
    track_running_stats : bool, optional
        If True, tracks the running mean and variance. Default: True.
    deterministic : bool, optional
        If True, uses deterministic forward pass. Default: False.
    num_mc_samples : int, optional
        Number of Monte Carlo samples for stochastic forward pass. Default: 20.
    seed : int, optional
        Seed for random number generation. Default: None.
    """

    __constants__ = [
        "track_running_stats",
        "momentum",
        "eps",
        "weight",
        "bias",
        "running_mean",
        "running_var",
        "num_batches_tracked",
        "num_features",
        "affine",
        "num_mc_samples",
    ]

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        deterministic=False,
        num_mc_samples=20,
        seed=None,
    ):
        super(_BayesBatchNormMF, self).__init__()

        # Initialize parameters
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        self.deterministic = deterministic
        self.num_mc_samples = num_mc_samples
        self.parallel_eval = False
        self.seed = seed

        # Initialize the random number generator
        self.generator = torch.Generator()
        # Set the seed if provided
        if seed is not None:
            self.generator.manual_seed(seed)

        # Initialize affine parameters if required
        if self.affine:
            self.weight_mu = Parameter(torch.Tensor(num_features))
            self.weight_psi = Parameter(torch.Tensor(num_features))
            self.bias_mu = Parameter(torch.Tensor(num_features))
            self.bias_psi = Parameter(torch.Tensor(num_features))
        else:
            self.register_parameter("weight_mu", None)
            self.register_parameter("weight_psi", None)
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_psi", None)

        # Initialize running stats if required
        if self.track_running_stats:
            self.register_buffer("running_mean", torch.zeros(num_features))
            self.register_buffer("running_var", torch.ones(num_features))
            self.register_buffer(
                "num_batches_tracked", torch.tensor(0, dtype=torch.long)
            )
        else:
            self.register_parameter("running_mean", None)
            self.register_parameter("running_var", None)
            self.register_parameter("num_batches_tracked", None)

        # Reset parameters
        self.reset_parameters()

        self.weight_size = list(self.weight_mu.shape) if self.affine else None
        self.bias_size = list(self.bias_mu.shape) if self.affine else None
        self.mul_exp_add = MulExpAddFunction.apply

        self._compute_exponential_factor()

    def reset_running_stats(self):
        """
        Reset the running mean and variance.
        """
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked.zero_()

    def reset_parameters(self):
        """
        Initialize the parameters (weights and biases) of the layer.
        """
        self.reset_running_stats()
        if self.affine:
            self.weight_mu.data.fill_(1)
            self.weight_psi.data.uniform_(-6, -5)
            self.bias_mu.data.zero_()
            self.bias_psi.data.uniform_(-6, -5)

    def _check_input_dim(self, input):
        """
        Check the dimensionality of the input.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to check.
        """
        raise NotImplementedError

    def _compute_exponential_factor(self):
        """
        Compute the exponential average factor for running stats.
        """
        if self.momentum is None:
            self.exponential_average_factor = 0.0
        else:
            self.exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self.momentum is None:
                    self.exponential_average_factor = 1.0 / float(
                        self.num_batches_tracked
                    )
                else:
                    self.exponential_average_factor = self.momentum

    def _forward_deterministic(self, input):
        """
        Forward pass for deterministic mode.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying batch normalization.
        """
        out = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training or not self.track_running_stats,
            self.exponential_average_factor,
            self.eps,
        )

        if self.affine:
            weight = self.weight_mu.unsqueeze(0)
            bias = self.bias_mu.unsqueeze(0)

            if out.dim() == 4:
                out = torch.addcmul(
                    bias[:, :, None, None], weight[:, :, None, None], out
                )
            elif out.dim() == 2:
                out = torch.addcmul(bias, weight, out)
            else:
                raise NotImplementedError
        return out

    def _forward_reparam_trick(self, input):
        """
        Forward pass using the reparameterization trick.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output tensor after applying batch normalization with reparameterization.
        """
        out = F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            None,
            None,
            self.training or not self.track_running_stats,
            self.exponential_average_factor,
            self.eps,
        )

        if self.affine:
            bs = input.shape[0]
            weight = self.mul_exp_add(
                torch.normal(
                    0, 1, size=(bs, *self.weight_size), device=input.device, dtype=input.dtype),  # Generate random perturbations
                self.weight_psi,
                self.weight_mu,
            )

            bias = self.mul_exp_add(
                torch.normal(
                    0, 1, size=(bs, *self.bias_size), device=input.device, dtype=input.dtype),  # Generate random perturbations
                self.bias_psi,
                self.bias_mu,
            )
            if out.dim() == 4:
                out = torch.addcmul(
                    bias[:, :, None, None], weight[:, :, None, None], out
                )
            elif out.dim() == 2:
                out = torch.addcmul(bias, weight, out)
            else:
                raise NotImplementedError
        return out

    def update_mode(self, mode):
        """
        Update the mode of operation for the layer.

        Parameters
        ----------
        mode : str
            Mode of operation. One of "deterministic", "reparam", "flipout".
        """
        self.mode = mode

        if mode == "deterministic":
            self.forward = self._forward_deterministic
        elif mode == "reparam":
            self.forward = self._forward_reparam_trick
        elif mode == "flipout":
            self.forward = self._forward_reparam_trick
        else:
            raise ValueError("Invalid mode")

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        Load the state dictionary for the layer.

        Parameters
        ----------
        state_dict : dict
            State dictionary to load from.
        prefix : str
            Prefix for the keys in the state dictionary.
        local_metadata : dict
            Local metadata.
        strict : bool
            Whether to strictly enforce that the keys in `state_dict` match the keys returned by `model.state_dict()`.
        missing_keys : list
            List of missing keys.
        unexpected_keys : list
            List of unexpected keys.
        error_msgs : list
            List of error messages.
        """
        version = local_metadata.get("version", None)

        if (version is None or version < 2) and self.track_running_stats:
            num_batches_tracked_key = prefix + "num_batches_tracked"
            if num_batches_tracked_key not in state_dict:
                state_dict[num_batches_tracked_key] = torch.tensor(0, dtype=torch.long)

        super(_BayesBatchNormMF, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )


class BayesBatchNorm2dMF(_BayesBatchNormMF):
    """
    Applies Bayesian Batch Normalization over a 2D input
    """

    def _check_input_dim(self, input):
        """
        Check the dimensionality of the input.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to check.
        """
        if input.dim() != 4 and input.dim() != 5:
            raise ValueError("expected 4D input (got {}D input)".format(input.dim()))


class BayesBatchNorm1dMF(_BayesBatchNormMF):
    """
    Applies Bayesian Batch Normalization over a 1D input
    """

    def _check_input_dim(self, input):
        """
        Check the dimensionality of the input.

        Parameters
        ----------
        input : torch.Tensor
            Input tensor to check.
        """
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError(
                "expected 2D or 3D input (got {}D input)".format(input.dim())
            )
