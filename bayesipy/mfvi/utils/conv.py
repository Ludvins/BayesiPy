import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
from .utils import MulExpAddFunction


class _BayesConvNdMF(Module):
    """
    Applies Bayesian Convolution

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple
        Stride of the convolution.
    padding : int or tuple
        Zero-padding added to both sides of the input.
    dilation : int or tuple
        Spacing between kernel elements.
    groups : int
        Number of blocked connections from input channels to output channels.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: True.
    seed : int, optional
        Seed for random number generation. Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        groups,
        bias,
        seed=None,
    ):
        super(_BayesConvNdMF, self).__init__()
        
        # Check if in_channels and out_channels are divisible by groups
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        
        # Initialize parameters
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        if isinstance(self.kernel_size, int):
            self.kernel_size = (self.kernel_size, self.kernel_size)
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.seed = seed

        # Initialize the random number generator
        self.generator = torch.Generator()
        # Set the seed if provided
        if seed is not None:
            self.generator.manual_seed(seed)

        # Initialize mean and log variance of weights
        self.weight_mu = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )
        self.weight_psi = Parameter(
            torch.Tensor(out_channels, in_channels // groups, *self.kernel_size)
        )

        # Initialize bias parameters if bias is True
        if bias is None or bias is False:
            self.bias = False
        else:
            self.bias = True

        if self.bias:
            self.bias_mu = Parameter(torch.Tensor(out_channels))
            self.bias_psi = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias_mu", None)
            self.register_parameter("bias_psi", None)

        # Reset parameters
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initializes the parameters (weights and biases) of the layer.
        """
        n = self.in_channels
        n *= np.prod(list(self.kernel_size))
        stdv = 1.0 / math.sqrt(n)
        
        # Initialize weights with uniform distribution
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_psi.data.uniform_(-6, -5)
        
        if self.bias:
            # Initialize biases with uniform distribution
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_psi.data.uniform_(-6, -5)

    def extra_repr(self):
        """
        Provides extra representation for printing.

        Returns
        -------
        str
            String representation of the layer.
        """
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        s += ", padding={padding}"
        s += ", dilation={dilation}"
        s += ", groups={groups}"
        s += ", bias={}".format(self.bias)
        s += ", seed={}".format(self.seed)
        return s.format(**self.__dict__)

    def __setstate__(self, state):
        """
        Set the state of the module.
        """
        super(_BayesConvNdMF, self).__setstate__(state)


class BayesConv2dMF(_BayesConvNdMF):
    """
    Applies Bayesian Convolution for 2D inputs

    Parameters
    ----------
    in_channels : int
        Number of channels in the input image.
    out_channels : int
        Number of channels produced by the convolution.
    kernel_size : int or tuple
        Size of the convolving kernel.
    stride : int or tuple, optional
        Stride of the convolution. Default: 1.
    padding : int or tuple, optional
        Zero-padding added to both sides of the input. Default: 0.
    dilation : int or tuple, optional
        Spacing between kernel elements. Default: 1.
    groups : int, optional
        Number of blocked connections from input channels to output channels. Default: 1.
    bias : bool, optional
        If True, adds a learnable bias to the output. Default: False.
    seed : int, optional
        Seed for random number generation. Default: None.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=False,
        seed=None,
    ):
        super(BayesConv2dMF, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            seed,
        )
        
        # Store weight shape
        self.weight_size = list(self.weight_mu.shape)
        # Store bias shape if bias is used
        self.bias_size = list(self.bias_mu.shape) if self.bias else None
        # Store the custom function
        self.mul_exp_add = MulExpAddFunction.apply

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
            Output tensor after applying the convolution.
        """
        return F.conv2d(
            x,
            weight=self.weight_mu,
            bias=self.bias_mu,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
            padding=self.padding,
        )

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
            Output tensor after applying the convolution with reparameterization.
        """
        # Batch size
        bs = x.shape[0]
        
        # Sample perturbations for weights
        weight = self.mul_exp_add(
            torch.normal(0, 1, size=(bs, *self.weight_size), device=x.device, dtype=x.dtype),  # Generate random perturbations
            self.weight_psi,
            self.weight_mu,
        ).view(bs * self.weight_size[0], *self.weight_size[1:])
        
        # Apply convolution
        out = F.conv2d(
            x.view(1, -1, x.shape[2], x.shape[3]),
            weight=weight,
            bias=None,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups * bs,
            padding=self.padding,
        )
        
        # Reshape output
        out = out.view(bs, self.out_channels, out.shape[2], out.shape[3])

        if self.bias:
            # Sample perturbations for biases
            bias = self.mul_exp_add(
                torch.normal(0, 1, size=(bs, *self.bias_size), device=x.device, dtype=x.dtype),  # Generate random perturbations
                self.bias_psi,
                self.bias_mu,
            )
            # Add bias to output
            out = out + bias[:, :, None, None]
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
            Output tensor after applying the convolution with reparameterization.
        """
        weight = torch.randn_like(self.weight_mu).mul_(self.weight_psi.exp()).add_(self.weight_mu)
        out = F.conv2d(input, weight=weight, bias=None,
                                stride=self.stride, dilation=self.dilation,
                                groups=self.groups, padding=self.padding)
        
        if self.bias:
            bias = torch.randn_like(self.bias_mu).mul_(self.bias_psi.exp()).add_(self.bias_mu)
        
            return out + bias
        
        return out
    


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
            Output tensor after applying the convolution with Flipout.
        """
        assert not self.bias
        
        # Apply convolution
        outputs = F.conv2d(
            x,
            weight=self.weight_mu,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

        # Sampling perturbation signs
        sign_input = (
            torch.empty(x.size(0), x.size(1), 1, 1, device=x.device)
            .uniform_(-1, 1)
            .sign()
        )
        sign_output = (
            torch.empty(
                outputs.size(0), outputs.size(1), 1, 1, device=x.device
            )
            .uniform_(-1, 1)
            .sign()
        )

        # Getting perturbation weights
        delta_kernel = torch.randn_like(self.weight_psi).mul(self.weight_psi.exp())

        # Perturbed feedforward
        perturbed_outputs = F.conv2d(
            x * sign_input,
            weight=delta_kernel,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        
        return outputs + perturbed_outputs * sign_output

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
            self.forward = self._forward_flipout
        else:
            raise ValueError("Invalid mode")
