import torch
from torch.nn import BatchNorm2d, Conv2d, Linear, PReLU

from .batchnorm import BayesBatchNorm2dMF
from .conv import BayesConv2dMF
from .linear import BayesLinearMF

def converter(input, psi_init_range=[-6, -5], seed=0):
    """
    Converts standard PyTorch layers to their Bayesian equivalents.

    Parameters
    ----------
    input : torch.nn.Module
        The input module (Linear, Conv2d, BatchNorm2d, or PReLU).
    psi_init_range : list of float, optional
        Range for initializing psi parameters. Default is [-6, -5].
    seed : int, optional
        Seed for random number generation. Default is 0.

    Returns
    -------
    torch.nn.Module
        The converted Bayesian module.
    """
    # Initialize the random number generator with the specified seed
    generator = torch.Generator()
    generator.manual_seed(seed)

    # Check if the input is one of the specific layer types we want to convert
    if isinstance(input, (Linear, Conv2d, BatchNorm2d, PReLU)):
        if isinstance(input, Linear):
            # Convert Linear layer to BayesLinearMF
            output = BayesLinearMF(
                input.in_features,
                input.out_features,
                input.bias,
                seed=seed,
            )
        elif isinstance(input, Conv2d):
            # Convert Conv2d layer to BayesConv2dMF
            output = BayesConv2dMF(
                input.in_channels,
                input.out_channels,
                input.kernel_size,
                input.stride,
                input.padding,
                input.dilation,
                input.groups,
                input.bias,
                seed=seed,
            )
        elif isinstance(input, BatchNorm2d):
            # Convert BatchNorm2d layer to BayesBatchNorm2dMF
            output = BayesBatchNorm2dMF(
                input.num_features,
                input.eps,
                input.momentum,
                input.affine,
                input.track_running_stats,
                seed=seed,
            )
            # Copy the running statistics
            output.running_mean = input.running_mean
            output.running_var = input.running_var
            output.num_batches_tracked = input.num_batches_tracked
        else:
            # Raise an error if the layer type is not supported
            raise NotImplementedError(f"Unsupported layer type: {type(input)}")

        # Copy the weights from the original layer if they exist
        if input.weight is not None:
            with torch.no_grad():
                output.weight_mu = input.weight

        # Copy the biases from the original layer if they exist
        if hasattr(input, "bias") and input.bias is not None:
            with torch.no_grad():
                output.bias_mu = input.bias

        # Initialize weight_psi with a uniform distribution using the generator
        if output.weight_psi is not None:
            output.weight_psi.data.uniform_(psi_init_range[0], psi_init_range[1], generator=generator)
        
        # Initialize bias_psi with a uniform distribution using the generator, if it exists
        if hasattr(output, "bias_psi") and output.bias_psi is not None:
            output.bias_psi.data.uniform_(psi_init_range[0], psi_init_range[1], generator=generator)
        
        # Delete the original input to save memory
        del input
        return output

    # If the input is a module container, recursively convert its children
    output = input
    for name, module in input.named_children():
        output.add_module(name, converter(module, psi_init_range, seed))
    
    # Delete the original input to save memory
    del input
    return output
