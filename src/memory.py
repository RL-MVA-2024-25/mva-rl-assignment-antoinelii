from typing import *
import os
os.environ['KMP_WARNINGS'] = 'off'
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from numba import njit
from segment_tree import SumSegmentTree, MinSegmentTree


class ReplayBuffer:
    '''A simple numpy replay buffer.'''
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
    ):
        self.obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.next_obs_buf = np.zeros((size, obs_dim), dtype=np.float32)
        self.acts_buf = np.zeros((size), dtype=np.float32)
        self.rews_buf = np.zeros((size), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr = 0
        self.size = 0
        
    def store(
        self, 
        obs: np.ndarray, 
        act: np.ndarray, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> None:
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
            done=self.done_buf[idxs],
        )

    def __len__(self) -> int:
        return self.size


class PrioritizedReplayBuffer(ReplayBuffer):
    '''Prioritized Replay buffer.'''
    def __init__(
        self, 
        obs_dim: int, 
        size: int, 
        batch_size: int = 32, 
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment_per_sampling: float = 0.000005,
    ):
        '''Initialization.'''
        assert alpha >= 0
        
        super().__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha
        self.beta = beta
        self.beta_increment_per_sampling = beta_increment_per_sampling
        
        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)
        
    def store(
        self, 
        obs: np.ndarray, 
        act: int, 
        rew: float, 
        next_obs: np.ndarray, 
        done: bool,
    ) -> None:
        '''Store experience and priority.'''
        super().store(obs, act, rew, next_obs, done)
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self) -> Dict[str, np.ndarray]:
        '''Sample a batch of experiences.'''
        assert len(self) >= self.batch_size
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])
        
        indices = self._sample_proportional()
        
        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        done = self.done_buf[indices]
        weights = self._calculate_weights(indices, self.beta)
        
        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            done=done,
            weights=weights,
            indices=indices,
        )
        
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        '''Update priorities of sampled transitions.'''
        assert len(indices) == len(priorities)
        _update_priorities_helper(indices, priorities, self.sum_tree, self.min_tree, self.alpha)
        self.max_priority = max(self.max_priority, priorities.max())
            
    def _sample_proportional(self) -> np.ndarray:
        '''Sample indices based on proportions.'''
        return _sample_proportional_helper(self.sum_tree, len(self), self.batch_size)
    
    def _calculate_weights(self, indices: np.ndarray, beta: float) -> np.ndarray:
        '''Calculate the weights of the experiences'''
        return _calculate_weights_helper(indices, beta, self.sum_tree, self.min_tree, len(self))


@njit(cache=True)
def _sample_proportional_helper(
    sum_tree: SumSegmentTree,
    size: int,
    batch_size: int,
) -> np.ndarray:
    indices = np.zeros(batch_size, dtype=np.int32) 
    p_total = sum_tree.sum(0, size - 1)
    segment = p_total / batch_size
    
    for i in range(batch_size):
        a = segment * i
        b = segment * (i + 1)
        upperbound = np.random.uniform(a, b)
        idx = sum_tree.retrieve(upperbound)
        indices[i] = idx
        
    return indices


@njit(cache=True)
def _calculate_weights_helper(
    indices: np.ndarray,
    beta: float,
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    size: int,
) -> np.ndarray:

    weights = np.zeros(len(indices), dtype=np.float32)

    for i in range(len(indices)):

        idx = indices[i]

        # get max weight
        p_min = min_tree.min() / sum_tree.sum()
        max_weight = (p_min * size) ** (-beta)
        
        # calculate weights
        p_sample = sum_tree[idx] / sum_tree.sum()
        weight = (p_sample * size) ** (-beta)
        weight = weight / max_weight
        
        weights[i] = weight
    
    return weights


@njit(cache=True)
def _update_priorities_helper(
    indices: np.ndarray,
    priorities: np.ndarray, 
    sum_tree: SumSegmentTree,
    min_tree: MinSegmentTree,
    alpha: float,
) -> None:

    for i in range(len(indices)):
        idx = indices[i]
        priority = priorities[i]
        sum_tree[idx] = priority ** alpha
        min_tree[idx] = priority ** alpha



def _gather_per_buffer_attr(memory: Optional[PrioritizedReplayBuffer]) -> dict:
    if memory is None:
        return {}
    per_buffer_keys = [
        'obs_buf', 'next_obs_buf', 'acts_buf', 'rews_buf', 'done_buf',
        'max_size', 'batch_size', 'ptr', 'size',
        'max_priority', 'tree_ptr', 'alpha',
    ]
    result = {key: getattr(memory, key) for key in per_buffer_keys}
    result['sum_tree'] = dict(
        capacity = memory.sum_tree.capacity,
        tree = memory.sum_tree.tree,
    )
    result['min_tree'] = dict(
        capacity = memory.min_tree.capacity,
        tree = memory.min_tree.tree,
    )
    return result