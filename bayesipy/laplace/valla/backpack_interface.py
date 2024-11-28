import torch
from backpack import backpack, extend, memory_cleanup
from backpack.context import CTX
from backpack.extensions import BatchGrad
from laplace.curvature import CurvatureInterface


class BackPackInterface(CurvatureInterface):
    """Interface for Backpack backend."""

    def __init__(self, model, output_dim):
        super().__init__(model, "regression", False, None)
        extend(self._model)
        self.output_size = output_dim

    def jacobians(self, x, enable_back_prop=False):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad per output dimension.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, input_shape)
            input data on compatible device with model.
        enable_back_prop : boolean
            If True, computational graph is retained and backpropagation can be used
            on elements used in this function.

        Returns
        -------
        Js : torch.Tensor of shape (batch, parameters, outputs)
            Jacobians ``
        """
        # Enable grads in this section of code
        with torch.set_grad_enabled(True):
            # Extend model using BackPack converter
            model = extend(self.model, use_converter=True)
            # Set model in evaluation mode to ignore Dropout, BatchNorm..
            model.eval()

            # Initialice array to concatenate
            to_stack = []

            # Loop over output dimension
            for i in range(self.output_size):
                # Reset gradients
                model.zero_grad()
                # Compute output
                out = model(x)

                # Use Backpack Gradbatch to retain independent gradients for each input.
                with backpack(BatchGrad()):
                    # Compute backward pass on the corresponding output (if more than one)
                    if self.output_size > 1:
                        out[:, i].sum().backward(
                            create_graph=enable_back_prop, retain_graph=enable_back_prop
                        )
                    else:
                        out.sum().backward(
                            create_graph=enable_back_prop, retain_graph=enable_back_prop
                        )
                    # Auxiliar array
                    to_cat = []
                    # Loop over model parameters, retrieve their gradient and delete it
                    for param in model.parameters():
                        to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                        delattr(param, "grad_batch")
                    # Stack all gradients
                    Jk = torch.cat(to_cat, dim=1)
                # Append result
                if i == 0:
                    to_stack = Jk.unsqueeze(0)
                else:
                    to_stack = torch.cat([to_stack, Jk.unsqueeze(0)], 0)
                # to_stack.append(Jk)

        # Clean model gradients
        model.zero_grad()
        # Erase gradients form input
        x.grad = None
        # Clean BackPak hooks
        CTX.remove_hooks()
        # Clean extended model
        _cleanup(model)
        # Return Jacobians
        if self.output_size > 1:
            return to_stack.permute(1, 0, 2)
        else:
            return Jk.unsqueeze(-1).transpose(1, 2)

    def jacobians_on_outputs(self, x, outputs, enable_back_prop=True):
        """Compute Jacobians \\(\\nabla_{\\theta} f(x;\\theta)\\) at current parameter \\(\\theta\\)
        using backpack's BatchGrad on specific output dimensions for each input.

        For example, if outputs[i] = [3, 4], the returned Jacobian at position i
        will have shape (num_params, 2), where [:, 0] will contain the Jacobians
        wrt the 3rd output and [:, 1] the Jacobians wrt the 4th output.

        Parameters
        ----------
        x : torch.Tensor of shape (batch, input_shape)
            input data on compatible device with model.
        outputs : torch.Tensor of shape (batch, n_outputs)
            Contains the outputs wrt which the Jacobian sholud be computed for every
            input.
        enable_back_prop : boolean
            If True, computational graph is retained and backpropagation can be used
            on elements used in this function.

        Returns
        -------
        Js : torch.Tensor
            Jacobians `(batch, parameters, n_outputs)`
            The position (i, p, j) contains the Jacobian wrt the p-th parameter
            for the input x[i] and the outputs[j]-th output of the network.
        """
        # Enable grads in this section of code
        with torch.set_grad_enabled(True):
            # Extend model using BackPack converter
            model = extend(self.model, use_converter=True)
            # Set model in evaluation mode to ignore Dropout, BatchNorm..
            model.eval()

            # Initialice array to concatenate
            to_stack = []

            # Loop over the number of desired outputs.
            for i in range(outputs.shape[1]):
                # Reset gradients
                model.zero_grad()
                # Compute output
                out = model(x)
                # Get the specific output for each input in this iteration
                c = outputs[:, i]

                # Use Backpack Gradbatch to retain independent gradients for each input.
                with backpack(BatchGrad()):
                    # Gather the desired output for each input
                    o = torch.gather(out, 1, c.unsqueeze(-1)).sum()
                    # Compute Backward pass
                    o.backward(
                        create_graph=enable_back_prop, retain_graph=enable_back_prop
                    )
                    # Initialize auxiliar array
                    to_cat = []
                    # Loop over model parameters, retrieve their gradient and delete it
                    for param in model.parameters():
                        to_cat.append(param.grad_batch.reshape(x.shape[0], -1))
                        delattr(param, "grad_batch")
                    # Stack all gradients
                    Jk = torch.cat(to_cat, dim=1)
                # Append result
                to_stack.append(Jk)

        # Clean model gradients
        model.zero_grad()
        # Erase gradients form input
        x.grad = None
        # Clean BackPak hooks
        CTX.remove_hooks()
        # Clean extended model
        _cleanup(model)

        # Return Jacobians
        if self.output_size > 1:
            return torch.stack(to_stack, dim=2).transpose(1, 2)
        else:
            return Jk.unsqueeze(-1).transpose(1, 2)


def _cleanup(module):
    for child in module.children():
        _cleanup(child)

    setattr(module, "_backpack_extend", False)
    memory_cleanup(module)
