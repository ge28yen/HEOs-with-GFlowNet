from typing import *

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType
import numpy as np

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong, tfloat

# States are 6-element-long lists of floats, where:
# - The floats sum up to 1 (e.g., representing probabilities or normalized values).
# - Each state element can be modified during the environment's operation.

class HEO(GFlowNetEnv):
    def __init__(self, max_length=6, max_value=1, pad_value=2, n_dim=2, n_angles=7, **kwargs):
        """
        Initializes the HEO environment.

        Args:
        -----
        max_length : int
            Maximum length of the state (number of elements in the state list).
        
        max_value : float
            Maximum possible value for elements in the state.

        pad_value : float
            Value used to pad unused positions in the state.

        n_dim : int
            Dimensionality of the action space (not directly used but reserved for extensions).

        n_angles : int
            A parameter (not explicitly utilized here, likely for an extended feature).

        kwargs : dict
            Additional arguments passed to the superclass.
        """
        self.sum = 1  # The sum constraint for state elements
        self.max_length = max_length
        self.max_value = max_value
        self.pad_value = pad_value
        self.n_dim = n_dim
        self.eos_token = np.float64(-1)  # Special token indicating end of sequence
        self.n_angles = n_angles
        self.length_traj = max_length - 1  # Maximum trajectory length
        self.done = False  # Indicates whether the episode is complete

        # Initialize the state with padding values
        self.source = [self.pad_value] * max_length
        super().__init__(**kwargs)

    def get_action_space(self):
        """
        Defines the continuous action space, including the end-of-sequence (EOS) token.
        
        Returns:
        --------
        A list of actions where each action is a tuple with one value.
        """
        first = list(np.linspace(0., 500, 101)) + [self.eos_token]
        return [(n,) for n in first]

    def step(self, action: List, skip_mask_check: bool = False):
        """
        Executes a step in the environment based on the given action.

        Args:
        -----
        action : tuple
            The action to apply. Each action modifies the state list.

        skip_mask_check : bool
            If True, skips validation of the action's validity.

        Returns:
        --------
        self.state : list
            The updated state after applying the action.

        action : tuple
            The action that was executed.

        valid : bool
            Indicates whether the action was valid.
        """
        do_step, self.state, action = self._pre_step(action, skip_mask_check)
        if not do_step:
            return self.state, action, False

        valid = True  # Default validity of the action

        # Mark episode as done if maximum length is reached or EOS token is encountered
        if self.n_actions == self.max_length or action[0] == self.eos_token:
            self.done = True
            return self.state, action, valid

        # Update the state based on the action
        self.state[self.n_actions] = action[0] / 1000  # Normalize action value
        self.n_actions += 1

        return self.state, action, valid

    def get_parents(self, state, done):
        """
        Identifies the parent state and action leading to the current state.

        Args:
        -----
        state : list
            The current state of the environment.

        done : bool
            Indicates whether the episode is complete.

        Returns:
        --------
        parents : list
            A list containing the parent state.

        actions : list
            A list of actions leading to the current state.
        """
        if done:
            return [state], [self.eos_token]
        if state == self.source:
            return [], []  # No parents for the source state
        else:
            # Revert the last action to identify the parent state
            for i in range(1, self.max_length + 1):
                if state[len(state) - i] == self.pad_value:
                    continue
                else:
                    action_number = state[len(state) - i]
                    state[len(state) - i] = self.pad_value
                    break
        return [state], [action_number * 1000]

    def get_mask_invalid_actions_forward(self, state: Optional[List] = None, done: Optional[bool] = None) -> List:
        """
        Creates a mask indicating invalid actions for the current state.

        Args:
        -----
        state : list
            The current state of the environment.

        done : bool
            Indicates whether the episode is complete.

        Returns:
        --------
        A list of booleans indicating whether each action is invalid.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True] * 102
        elif state[-1] >= self.length_traj:
            return [True] * 102
        else:
            return [False] * 102

    def states2policy(self, states: Union[List[List[int]], List[TensorType["max_length"]]]) -> TensorType["batch", "policy_input_dim"]:
        """
        Converts a batch of states into a format suitable for the policy model.

        Each state is one-hot encoded.

        Args:
        -----
        states : list or tensor
            A batch of states.

        Returns:
        --------
        A tensor of one-hot encoded states.
        """
        states = tlong(states, device=self.device)
        return F.one_hot(states, 102).reshape(states.shape[0], -1).to(self.float)

    def states2proxy(self, states: Union[List[List[int]], List[TensorType["max_length"]]]) -> TensorType["batch", "state_dim"]:
        """
        Converts a batch of states into a format suitable for the proxy model.

        Args:
        -----
        states : list or tensor
            A batch of states.

        Returns:
        --------
        A tensor representation of the states.
        """
        return tfloat(states, device=self.device, float_type=self.float)

    def state2readable(self, state: List[int] = None) -> str:
        """
        Converts a state into a human-readable string.

        Args:
        -----
        state : list
            A state represented as a list of integers.

        Returns:
        --------
        A space-separated string representation of the state.
        """
        if state is None:
            state = self.state
        return " ".join(str(x) for x in state)

    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a human-readable string back into a state.

        Args:
        -----
        readable : str
            A space-separated string representation of a state.

        Returns:
        --------
        A state represented as a list of integers.
        """
        return [int(x) for x in readable.split()]