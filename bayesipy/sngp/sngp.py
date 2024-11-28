import torch
import math
import copy
from torch import nn
from typing import Optional
from torch import Tensor
from torch.nn.utils.parametrizations import spectral_norm
from tqdm import tqdm
from torch.nn import init

def RandomFeatureLinear(i_dim: int, o_dim: int, bias: bool = True, require_grad: bool = False, device: Optional[torch.device] = None, dtype: Optional[torch.dtype] = None) -> nn.Linear:
    """
    Creates a linear layer with randomized weights and biases.

    Parameters
    ----------
    i_dim : int
        Input dimension of the linear layer.
    o_dim : int
        Output dimension of the linear layer.
    bias : bool, optional
        Whether to include a bias term. Default is True.
    require_grad : bool, optional
        Whether the parameters require gradients. Default is False.
    device : torch.device, optional
        The device to which the layer should be moved (e.g., 'cpu' or 'cuda'). Default is None.
    dtype : torch.dtype, optional
        The data type of the parameters. Default is None.

    Returns
    -------
    nn.Linear
        The linear layer with randomized weights and biases.
    """
    # Create a linear layer
    m = nn.Linear(i_dim, o_dim, bias).to(device).to(dtype)
    
    # Initialize weights with normal distribution
    nn.init.normal_(m.weight, mean=0.0, std=0.05)
    
    # Set requires_grad for weights
    m.weight.requires_grad = require_grad
    
    if bias:
        # Initialize bias with uniform distribution
        nn.init.uniform_(m.bias, a=0.0, b=2. * math.pi)
        # Set requires_grad for bias
        m.bias.requires_grad = require_grad
    
    return m

class SNGP(nn.Module):
    """
    A PyTorch module implementing the Stochastic Neural Gaussian Process (SNGP) model.

    Parameters
    ----------
    model : nn.Module
        The base model to which SNGP is applied.
    gp_kernel_scale : float, optional
        Scale of the GP kernel. Default is 1.0.
    n_random_features : int, optional
        Number of random features. Default is 1024.
    gp_output_bias : float, optional
        Bias for the GP output. Default is 0.0.
    layer_norm_eps : float, optional
        Epsilon value for layer normalization. Default is 1e-12.
    n_power_iterations : int, optional
        Number of power iterations for spectral normalization. Default is 1.
    scale_random_features : bool, optional
        Whether to scale random features. Default is True.
    normalize_input : bool, optional
        Whether to normalize input features. Default is True.
    gp_cov_momentum : float, optional
        Momentum for GP covariance updates. Default is 0.999.
    gp_cov_ridge_penalty : float, optional
        Ridge penalty for GP covariance. Default is 1e-3.
    seed : Optional[int], optional
        Random seed for reproducibility. Default is None.
    """
    
    def __init__(self, model: nn.Module, gp_kernel_scale: float = 1.0, n_random_features: int = 1024, gp_output_bias: float = 0.0, layer_norm_eps: float = 1e-12, n_power_iterations: int = 1, scale_random_features: bool = True, normalize_input: bool = True, gp_cov_momentum: float = 0.999, gp_cov_ridge_penalty: float = 1e-3, seed: Optional[int] = None):
        super(SNGP, self).__init__()
        
        # Set random seed if provided
        if seed is not None:
            torch.manual_seed(seed)
        
        p = list(model.parameters())[0]

        # Store device and dtype
        self.device = p.device
        self.dtype = p.dtype
        
        # Initialize model and hyperparameters
        self.model = model
        self.n_random_features = n_random_features
        self.layer_norm_eps = layer_norm_eps
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_momentum = gp_cov_momentum
        self.n_power_iterations = n_power_iterations
        
        # Calculate scaling factors
        self.gp_input_scale = 1. / math.sqrt(gp_kernel_scale)
        self.gp_feature_scale = math.sqrt(2. / float(n_random_features))
        self.gp_output_bias = gp_output_bias
        self.scale_random_features = scale_random_features
        self.normalize_input = normalize_input
        
        # Apply spectral normalization
        self.spectral_norm()
        
    def spectral_norm(self):
        """
        Apply spectral normalization to the linear and convolutional layers in the model.
        """
        for model in self.model.children():
            if isinstance(model, nn.Linear) or isinstance(model, nn.Conv2d):
                model = spectral_norm(model, n_power_iterations=self.n_power_iterations)
                
    def fit(self, train_loader: torch.utils.data.DataLoader, iterations: int, lr: float, weight_decay: float, verbose: bool = False) -> list:
        """
        Train the SNGP model.

        Parameters
        ----------
        train_loader : torch.utils.data.DataLoader
            DataLoader for the training data.
        iterations : int
            Number of training iterations.
        lr : float
            Learning rate for the optimizer.
        weight_decay : float
            Weight decay for the optimizer.
        verbose : bool, optional
            Whether to display a progress bar. Default is False.

        Returns
        -------
        list
            List of loss values recorded during training.
        """
        # Extract a batch of data from the train_loader
        data = next(iter(train_loader))
        X = data[0]
        
        # Check output dimension
        try:
            out = self.model(X[:1].to(self.device).to(self.dtype))
        except (TypeError, AttributeError):
            out = self.model(X.to(self.device).to(self.dtype))

        # Store the output dimension of the model
        self.n_outputs = out.shape[-1]
        
        # Replace the model's last layer with an identity layer
        self.model.fc = nn.Identity()
        try:
            out = self.model(X[:1].to(self.device).to(self.dtype))
        except (TypeError, AttributeError):
            out = self.model(X.to(self.device).to(self.dtype))
        self.hidden_size = out.shape[-1]
        
        # Initialize random feature layer
        self._random_feature = RandomFeatureLinear(self.hidden_size, self.n_random_features, device=self.device, dtype=self.dtype)
        # Initialize normalization layer
        self._gp_input_normalize_layer = torch.nn.LayerNorm(self.hidden_size, eps=self.layer_norm_eps).to(self.device).to(self.dtype)   
        # Initialize GP output bias
        self._gp_output_bias = torch.tensor([self.gp_output_bias] * self.n_outputs).to(self.device).to(self.dtype)
        
        # Initialize beta parameter
        self._beta = torch.Tensor(self.n_random_features, self.n_outputs).to(self.device).to(self.dtype)
        init.kaiming_uniform_(self._beta, a=math.sqrt(5))
        # Convert beta to parameter
        self._beta = torch.nn.Parameter(self._beta, requires_grad=True)

        # Initialize precision matrix
        self.initial_precision_matrix = self.gp_cov_ridge_penalty * torch.eye(self.n_random_features).to(self.device).to(self.dtype)
        self.precision_matrix = copy.deepcopy(self.initial_precision_matrix)
        
        # Initialize optimizer
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Initialize progress bar if verbose
        if verbose:
            iters = tqdm(range(iterations), unit=" iteration")
            iters.set_description("Training ")
        else:
            iters = range(iterations)
        
        # Initialize data iterator
        data_iter = iter(train_loader)
        
        losses = []
        for i in iters:
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                # Reinitialize data loader if dataset ends
                data_iter = iter(train_loader)
                inputs, targets = next(data_iter)
                
            # Move inputs and targets to device
            inputs = inputs.to(self.device).to(self.dtype)
            targets = targets.to(self.device)
            
            # Forward pass
            logit = self.forward(inputs)

            # Compute and record loss
            loss = torch.nn.functional.cross_entropy(logit, targets)
            losses.append(loss.item())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute inverse of the precision matrix
        self.precision_matrix_inv = torch.linalg.inv(self.precision_matrix)

        return losses

    def gp_layer(self, gp_inputs: Tensor) -> (Tensor, Tensor):
        """
        Apply the GP layer transformation.

        Parameters
        ----------
        gp_inputs : Tensor
            The input features for the GP layer.

        Returns
        -------
        Tuple[Tensor, Tensor]
            Tuple containing the GP feature and GP output.
        """
        # Normalize input if required
        if self.normalize_input:
            gp_inputs = self._gp_input_normalize_layer(gp_inputs)

        # Apply random feature transformation
        gp_feature = self._random_feature(gp_inputs)
        # Apply cosine activation
        gp_feature = torch.cos(gp_feature)

        # Scale random features if required
        if self.scale_random_features:
            gp_feature = gp_feature * self.gp_input_scale

        # Compute GP output
        gp_output = torch.matmul(gp_feature, self._beta) + self._gp_output_bias

        if self.training:
            # Update precision matrix during training
            self.update_cov(gp_feature)
        
        return gp_feature, gp_output

    def reset_cov(self):
        """
        Reset the covariance matrix to its initial state.
        """
        self.precision_matrix = copy.deepcopy(self.initial_precision_matrix)

    @torch.no_grad()
    def update_cov(self, gp_feature: Tensor):
        """
        Update the covariance matrix using the GP features.

        Parameters
        ----------
        gp_feature : Tensor
            The features from the GP layer.
        """
        # Compute batch-wise precision matrix
        batch_size = gp_feature.size()[0]
        precision_matrix_minibatch = torch.matmul(gp_feature.t(), gp_feature)
        
        # Update covariance matrix with momentum
        if self.gp_cov_momentum > 0:
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                self.gp_cov_momentum * self.precision_matrix +
                (1. - self.gp_cov_momentum) * precision_matrix_minibatch
            )
        else:
            precision_matrix_new = self.precision_matrix + precision_matrix_minibatch
        
        # Update precision matrix parameter
        self.precision_matrix = torch.nn.Parameter(precision_matrix_new, requires_grad=False)
    
    @torch.no_grad()
    def compute_predictive_covariance(self, gp_feature: Tensor) -> Tensor:
        """
        Compute the predictive covariance matrix.

        Parameters
        ----------
        gp_feature : Tensor
            The features from the GP layer.

        Returns
        -------
        Tensor
            The predictive covariance matrix.
        """
        # Compute feature covariance matrix
        feature_cov_matrix = torch.linalg.inv(self.precision_matrix)

        # Compute covariance matrix of the GP prediction
        cov_feature_product = torch.matmul(feature_cov_matrix, gp_feature.t()) * self.gp_cov_ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        
        return gp_cov_matrix

    @torch.no_grad()
    def mean_field_logits(self, logits: Tensor, covmat: Tensor) -> Tensor:
        """
        Adjust the logits to approximate the posterior mean using mean-field approximation.

        Parameters
        ----------
        logits : Tensor
            The logits from the model.
        covmat : Tensor
            The covariance matrix of the predictions.

        Returns
        -------
        Tensor
            The adjusted logits.
        """
        # Compute standard deviation from diagonal of covariance matrix
        variances = torch.diagonal(covmat)
        logits_scale = torch.sqrt(1. + variances)

        # Adjust logits scale to match dimensions
        if len(logits.shape) > 1:
            logits_scale = torch.unsqueeze(logits_scale, dim=-1)

        return logits / logits_scale

    @torch.no_grad()
    def monte_carlo_softmax(self, logits: Tensor, var: Tensor, num_samples: int = 50, temp_scale: float = 1.0) -> Tensor:
        """
        Estimate the softmax mean using Monte Carlo sampling.

        Parameters
        ----------
        logits : Tensor
            The logits from the model.
        var : Tensor
            The variance for each sample.
        num_samples : int, optional
            Number of Monte Carlo samples. Default is 50.
        temp_scale : float, optional
            Temperature scaling factor. Default is 1.0.

        Returns
        -------
        Tensor
            The estimated softmax probabilities.
        """
        var = torch.diag(var)
        stddev = torch.sqrt(var * temp_scale)
        shape = tuple(list(logits.shape) + [num_samples])
        
        # Sample from standard normal distribution
        rand_samples = torch.randn(shape).to(self.device).to(self.dtype)
        means = torch.tile(logits.unsqueeze(-1), (1, 1, num_samples))
        stddevs = torch.tile(stddev.unsqueeze(-1).unsqueeze(-1), (1, means.shape[1], num_samples))
        
        # Compute sampled logits
        sampled_logits = means + stddevs * rand_samples
        return sampled_logits.permute(2, 0, 1)

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass through the SNGP model.

        Parameters
        ----------
        x : Tensor
            Input tensor to the model.

        Returns
        -------
        Tensor
            Output logits of the model.
        """
        latent_feature = self.model(x)
        _, gp_output = self.gp_layer(latent_feature)
        return gp_output

    def predict(self, x: Tensor) -> Tensor:
        """
        Predict output for the input data.

        Parameters
        ----------
        x : Tensor
            Input tensor to the model.

        Returns
        -------
        Tensor
            Predicted output logits.
        """
        with torch.no_grad():
            latent_feature = self.model(x)
            gp_feature, gp_output = self.gp_layer(latent_feature)
            gp_cov_matrix = self.compute_predictive_covariance(gp_feature)
            gp_output = self.monte_carlo_softmax(gp_output, gp_cov_matrix)
        return gp_output
