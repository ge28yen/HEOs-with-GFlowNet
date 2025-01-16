from typing import *

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchtyping import TensorType
import numpy as np

from gflownet.envs.base import GFlowNetEnv
from gflownet.utils.common import copy, tlong, tfloat

#States are 6-element-long lists of floats.
#The floats should sum up to 1. Thus, each of them should be

class HEO(GFlowNetEnv):
    def __init__(self, max_length = 6, max_value = 1, pad_value = 2, n_dim = 2, n_angles = 7,**kwargs):
        self.sum = 1
        self.max_length = max_length
        self.max_value = max_value
        self.pad_value = pad_value
        self.n_dim = n_dim
        self.eos_token =  np.float64(-1)
        self.n_angles = n_angles # IM NOT SURE WHAT THIS IS
        self.length_traj = max_length - 1
        self.done = False

        self.source = [self.pad_value] * max_length
        super().__init__(**kwargs)

    def get_action_space(self):
        first = list(np.linspace(0.,500,101)) + [self.eos_token]
        return [(n, ) for n in first]
    
    
    def step(self, action: List, skip_mask_check: bool = False):
        """
        Executes step given an action.

        Args
        ----
        action : tuple
            Action to be executed. An action is a tuple int values indicating the
            dimensions to increment by 1.

        skip_mask_check : bool
            If True, skip computing forward mask of invalid actions to check if the
            action is valid.

        Returns
        -------
        self.state : list
            The sequence after executing the action

        action : tuple
            Action executed

        valid : bool
            False, if the action is not allowed for the current state.
        """
        do_step, self.state, action = self._pre_step(
            action, skip_mask_check #or skip_mask_step
        )
        if not do_step:
            return self.state, action, False
        
        valid = True
        

        if self.n_actions == self.max_length:
            self.done = True
            return self.state, action, valid
        elif action[0] == self.eos_token:
            self.done = True
            return self.state, action, valid
        
        #Update the state
        self.state[self.n_actions] = action[0]/1000
        self.n_actions +=1

        return self.state, action, valid
    
    def get_parents(self, state, done ):
        """
        Determines all parents and actions that lead to state.

        The GFlowNet graph is a tree and there is only one parent per state.

        Args
        ----
        state : tensor
            Input state. If None, self.state is used.

        done : bool
            Whether the trajectory is done. If None, self.done is used.

        action : None
            Ignored

        Returns
        -------
        parents : list
            List of parents in state format. This environment has a single parent per
            state.

        actions : list
            List of actions that lead to state for each parent in parents. This
            environment has a single parent per state.
        """
        if done:
            return [state], [self.eos_token]
        if state == self.source:
            return [], []
        else:
            # set the last element not equal to pad_value to pad_value
            for i in range(1, self.max_length+1):
                if state[len(state)-i] == self.pad_value:
                    continue
                else:
                    action_number = state[len(state)-i]
                    state[len(state)-i] = self.pad_value
                    break
        return [state], [action_number*1000]

    def get_mask_invalid_actions_forward(
        self,
        state: Optional[List] = None,
        done: Optional[bool] = None,
        ) -> List:
        """
        The action space is continuous, thus the mask is not of invalid actions as
        in discrete environments, but an indicator of "special cases", for example
        states from which only certain actions are possible.

        The "mask" has 2 elements - to match the mask of backward actions - but only
        one is needed for forward actions, thus both elements take the same value,
        according to the following:

        - If done is True, then the mask is True.
        - If the number of actions (state[-1]) is equal to the (fixed) trajectory
          length, then only EOS is valid and the mask is True.
        - Otherwise, any continuous action is valid (except EOS) and the mask is False.
        """
        if state is None:
            state = self.state.copy()
        if done is None:
            done = self.done
        if done:
            return [True] *102
        elif state[-1] >= self.length_traj:
            return [True] *102
        else:
            return [False] *102

    def states2policy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "policy_input_dim"]:
        """
        Prepares a batch of states in "environment format" for the policy model: states
        are one-hot encoded.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A tensor containing all the states in the batch.
        """
        states = tlong(states, device=self.device)
        return (
            F.one_hot(states, 102)
            .reshape(states.shape[0], -1)
            .to(self.float)
        )
    
    def states2proxy(
        self, states: Union[List[List[int]], List[TensorType["max_length"]]]
    ) -> TensorType["batch", "state_dim"]:
        """
        Prepares a batch of states in "environment format" for a proxy: the batch is
        simply converted into a tensor of indices.

        Args
        ----
        states : list or tensor
            A batch of states in environment format, either as a list of states or as a
            list of tensors.

        Returns
        -------
        A list containing all the states in the batch, represented themselves as lists.
        """
        return tfloat(states, device=self.device, float_type = self.float)
    def state2readable(self, state: List[int] = None) -> str:
        """
        Converts a state (list of ints) into a human-readable string: each integer
        is turned into its string form, and they're joined by spaces.

        Args
        ----
        state : list[int]
            A state in environment format. If None, self.state is used.

        Returns
        -------
        A string of space-separated integer values.
        """
        if state is None:
            state = self.state
        
        return " ".join(str(x) for x in state)
    
    def readable2state(self, readable: str) -> List[int]:
        """
        Converts a space-separated string of integers into a list of ints.

        Args
        ----
        readable : str
            A state in readable format - space-separated integers.

        Returns
        -------
        A list of integers.
        """
        return [int(x) for x in readable.split()]