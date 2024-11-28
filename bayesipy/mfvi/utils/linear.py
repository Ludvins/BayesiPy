import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
import math
from .utils import MulExpAddFunction


class BayesLinearMF(Module):
    """
    Applies a Bayesian Linear transformation to the incoming data.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    bias : bool, optional
        If set to False, the layer will not learn an additive bias. Default: True.
    mode : str, optional
        Mode of operation. Can be 'deterministic', 'reparam', or 'flipout'. Default: 'flipout'.
    seed : int, optional
        Seed for random number generation. Default: None.
    """
    def __init__(self, in_features, out_features, bias=True, mode="flipout", seed=None):
        super(BayesLinearMF, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = seed

        # Initialize the random number generator
        self.generator = torch.Generator()
        # Set the seed if provided
        if seed is not None:
            self.generator.manual_seed(seed)

        # Set the mode of operation
        self.update_mode(mode)

        # Initialize mean and log variance of weights
        self.weight_mu = Parameter(torch.Tensor(out_features, in_features))
        self.weight_psi = Parameter(torch.Tensor(out_features, in_features))

        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            # Initialize mean and log variance of biases
            self.bias_mu = Parameter(torch.Tensor(out_features))
            self.bias_psi = Parameter(torch.Tensor(out_features))
        else:
            # Register dummy parameters
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_psi", None)

        # Reset parameters to initial values
        self.reset_parameters()

        # Store weight shape
        self.weight_size = list(self.weight_mu.shape)
        # Store bias shape if bias is used
        self.bias_size = list(self.bias_mu.shape) if self.bias else None
        # Store the custom function
        self.mul_exp_add = MulExpAddFunction.apply

    def reset_parameters(self):
        """
        Initializes the parameters (weights and biases) of the layer.
        """
        stdv = 1.0 / math.sqrt(self.weight_mu.size(1))
        # Initialize weights with uniform distribution
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_psi.data.uniform_(-6, -5)
        if self.bias:
            # Initialize biases with uniform distribution
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_psi.data.uniform_(-6, -5)

    def _forward_deterministic(self, x):
        """
        Forward pass for deterministic mode.

        Parameters
        ----------
        x : torch.Tensor
            Contains the input tensor of the data.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation.
        """
        weight = self.weight_mu
        bias = self.bias_mu if self.bias else None
        # Apply the linear transformation
        out = F.linear(x, weight, bias)
        return out

    def _forward_reparam_trick(self, x):
        """
        Forward pass using the reparameterization trick.

        Parameters
        ----------
        x : torch.Tensor
            Contains the input tensor of the data.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation with reparameterization.
        """
        # Batch size
        bs = x.shape[0]
        # Sample perturbations for weights
        weight = self.mul_exp_add(
            torch.normal(
                0,
                1,
                size=(bs, *self.weight_size),
                device=x.device,
                dtype=x.dtype,
            ),  # Generate random perturbations
            self.weight_psi,
            self.weight_mu,
        )

        # Batch matrix multiplication
        out = torch.bmm(weight, x.unsqueeze(2)).squeeze(-1)
        if self.bias:
            # Sample perturbations for biases
            bias = self.mul_exp_add(
                torch.normal(
                    0,
                    1,
                    size=(bs, *self.bias_size),
                    device=x.device,
                    dtype=x.dtype,
                ),  # Generate random perturbations
                self.bias_psi,
                self.bias_mu,
            )
            # Add bias to output
            out = out + bias
        return out
    
    def _forward_mc(self, x):
        """
        Forward pass using Monte Carlo sampling.
        
        Parameters
        ----------
        x : torch.Tensor
            Contains the input tensor of the data.
            
        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation with Monte Carlo sampling.
        """
        weight = torch.randn_like(self.weight_mu).mul_(self.weight_psi.exp()).add_(self.weight_mu)
        if self.bias:
            bias = torch.randn_like(self.bias_mu).mul_(self.bias_psi.exp()).add_(self.bias_mu)
        else:
            bias = None
        return F.linear(x, weight, bias)
    
        

    def _forward_flipout(self, x):
        """
        Forward pass using the Flipout method.

        Parameters
        ----------
        x : torch.Tensor
            Contains the input tensor of the data.

        Returns
        -------
        torch.Tensor
            Output tensor after applying the linear transformation with Flipout.
        """
        outputs = F.linear(x, self.weight_mu, self.bias_mu if self.bias else None)
        # Sampling perturbation signs
        sign_input = (
            torch.empty_like(x).uniform_(-1, 1).sign()
        )
        sign_output = (
            torch.empty_like(outputs).uniform_(-1, 1).sign()
        )
        # Getting perturbation weights
        delta_kernel = torch.randn_like(self.weight_psi).mul(
            self.weight_psi.exp()
        )
        delta_bias = (
            torch.randn_like(self.bias_psi).mul(
                self.bias_psi.exp()
            )
            if self.bias
            else None
        )
        # Perturbed feedforward
        perturbed_outputs = F.linear(x * sign_input, delta_kernel, delta_bias)
        # Add perturbed outputs
        out = outputs + perturbed_outputs * sign_output
        return out

    def update_mode(self, mode):
        """
        Updates the mode of operation for the layer.

        Parameters
        ----------
        mode : str
            Mode of operation. Can be 'deterministic', 'reparam', or 'flipout'.

        Raises
        ------
        ValueError
            If an invalid mode is provided.
        """
        self.mode = mode

        if mode == "deterministic":
            self.forward = self._forward_deterministic
        elif mode == "reparam":
            self.forward = self._forward_reparam_trick
        elif mode == "flipout":
            self.forward = self._forward_flipout
        elif mode == "mc":
            self.forward = self._forward_mc
        else:
            raise ValueError("Invalid mode")