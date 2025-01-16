from typing import List, Union

import torch
import torch.nn as nn
from torchtyping import TensorType

from gflownet.proxy.base import Proxy
from gflownet.utils.common import tfloat

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Single output for regression
        )

    def forward(self, x):
        return self.model(x)

class HeoScorer(Proxy):
    """
    A 'dummy' scorer proxy that ignores the input and always
    outputs a tensor of constant values, matching the input batch size.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def __call__(
        self,
        states: Union[List[str], List[List[str]], TensorType["batch", "state_dim"]]
    ) -> TensorType["batch"]:
        """
        Args
        ----
        states : (1) Tensor of shape [batch, state_dim], or
                 (2) List of strings (each string is a 'word'), or
                 (3) List of list of strings (each sub-list is a tokenized word).

        Returns
        -------
        A 1D torch.Tensor of shape [batch], filled with a constant value (e.g. 42.0).
        """

        if torch.is_tensor(states):
            # states is a tensor of shape [batch, state_dim]
            batch_size = states.shape[0]
        elif isinstance(states, list):
            # states is a list of length 'batch'
            batch_size = len(states)
        else:
            raise NotImplementedError(
                "HeoScorer only supports a 2D tensor or list input."
            )
        model = MLP(input_size=6)
        model.load_state_dict(torch.load('regression_heo.pth', map_location=torch.device('cpu')))
        proxies = []

        for i in range(batch_size):
            zeroed_states = states[i]
            zeroed_states[zeroed_states == 2] = 1
            proxy = model(zeroed_states).item()
            proxies.append(proxy)

        # Return a 1D tensor of length [batch_size]
        # Here we just fill it with the constant 42.0.
        output = tfloat(
            proxies,
            float_type=self.float,  # Ensures the correct float precision
            device=self.device      # Ensures it lives on the chosen device
        )
        
        return output