from gym import Space
from gym.spaces import Box, Dict
import numpy as np
from typing import Union, SupportsFloat


class GraphSpace(Space):
    """A space supporting graph representations with a variable number of nodes and edges."""

    def __init__(self,
                 num_node_features: int,
                 low: Union[SupportsFloat, np.ndarray],
                 high: Union[SupportsFloat, np.ndarray],
                 dtype=np.float32,
                 seed=None):
        assert num_node_features > 0, "num_node_features have to be positive."
        assert dtype is not None, "dtype must be explicitly provided."
        self.dtype = np.dtype(dtype)
        self.num_node_features = num_node_features
        self.low = low
        self.high = high
        super(GraphSpace, self).__init__(None, self.dtype, seed)

    def sample(self):
        """Returns a dict"""
        num_nodes = np.floor(self.np_random.exponential())
        num_edges = np.floor(self.np_random.exponential())

        return Dict({
            "x": Box(low=self.low, high=self.high, shape=(num_nodes, self.num_node_features), dtype=self.dtype),
            "edge_index": Box(low=0, high=(num_nodes - 1), shape=(2, num_edges), dtype=np.int32)
        })

    def contains(self, x) -> bool:
        return True
        # return super().contains(x)

    def __eq__(self, other):
        return (self.dtype == other.dtype) and (self.num_node_features == other.num_node_features)
