from typing import List, Union

import torch
import torch.nn as nn
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) with the following structure:
    - Input layer with size `input_size`
    - Hidden layer with 64 units and ReLU activation
    - Hidden layer with 32 units and ReLU activation
    - Output layer with a single unit (regression output)
    """

    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),  # Input layer to hidden layer
            nn.ReLU(),                 # Activation for first hidden layer
            nn.Linear(64, 32),         # Hidden layer to second hidden layer
            nn.ReLU(),                 # Activation for second hidden layer
            nn.Linear(32, 1)           # Output layer for regression
        )

    def forward(self, x):
        """
        Forward pass through the MLP.
        Args:
        -----
        x: Input tensor.

        Returns:
        --------
        Output tensor (single regression value per input).
        """
        return self.model(x)

class HeoScorer(Proxy):
    """
    A scorer proxy that processes input states and calculates a proxy score
    using a pretrained MLP model. This implementation is a placeholder 
    ('dummy') scorer that demonstrates proxy computation while returning 
    outputs based on input states.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        states: Union[List[str], List[List[str]], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        """
        Process input states and compute proxy scores.

        Args:
        -----
        states : 
            (1) Tensor of shape [batch, state_dim], or
            (2) List of strings (each string is a 'word'), or
            (3) List of list of strings (each sub-list is a tokenized word).

        Returns:
        --------
        torch.Tensor: A 1D tensor of shape [batch] containing computed scores.
        """

        if torch.is_tensor(states):
            # If the input is a tensor, determine batch size from the tensor shape
            batch_size = states.shape[0]
        elif isinstance(states, list):
            # If the input is a list, determine batch size from the list length
            batch_size = len(states)
        else:
            # Raise an error for unsupported input types
            raise NotImplementedError(
                "HeoScorer only supports a 2D tensor or list input."
            )
        
        # Initialize the MLP model with a predefined input size
        model = MLP(input_size=6)
        # Load pretrained weights into the model
        model.load_state_dict(torch.load('regression_heo.pth', map_location=torch.device('cpu')))
        
        proxies = []  # To store computed proxy scores for each input

        for i in range(batch_size):
            # Process each input in the batch
            zeroed_states = states[i]
            zeroed_states[zeroed_states == 2] = 1  # Replace value '2' with '1' in the input
            proxy = model(zeroed_states).item()   # Compute proxy score using the model
            proxies.append(proxy)

        # Convert the list of proxy scores to a tensor with appropriate device and type
        output = tfloat(
            proxies,
            float_type=self.float,  # Ensure the correct float precision
            device=self.device      # Ensure tensor is on the chosen device
        )
        
        return output