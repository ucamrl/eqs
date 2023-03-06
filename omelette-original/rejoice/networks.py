from typing import Union, Tuple

import torch
from torch import nn
from torch.distributions import Categorical, Normal
import torch_geometric as pyg
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn.models.basic_gnn import BasicGNN
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear, ModuleList

from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    GINConv,
    MessagePassing,
    PNAConv,
    SAGEConv,
)
from torch_geometric.nn.models import MLP
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.typing import Adj


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class GATNetwork(nn.Module):
    """A Graph Attentional Network (GAT) which serves as a Policy Network for our RL agent."""

    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 3, hidden_size: int = 128,
                 out_std=np.sqrt(2), dropout=0.0, use_edge_attr=True):
        super(GATNetwork, self).__init__()
        self.use_edge_attr = use_edge_attr
        self.gnn = GAT(in_channels=num_node_features,
                       hidden_channels=hidden_size,
                       out_channels=hidden_size,
                       num_layers=n_layers,
                       add_self_loops=False,
                       dropout=dropout,
                       norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                       act="leaky_relu",
                       v2=True,
                       edge_dim=(2 if self.use_edge_attr else None))

        if dropout == 0.0:
            self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
                                  layer_init(nn.Linear(hidden_size, n_actions), std=out_std))
        else:
            self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
                                  nn.Linear(hidden_size, n_actions))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index,
                     edge_attr=(data.edge_attr if self.use_edge_attr else None))
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        x = self.head(x)
        return x


class GINNetwork(nn.Module):

    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 3, hidden_size: int = 128,
                 out_std=np.sqrt(2)):
        super(GINNetwork, self).__init__()
        self.gnn = pyg.nn.GIN(in_channels=num_node_features, hidden_channels=hidden_size,
                              out_channels=hidden_size, num_layers=n_layers,
                              norm=pyg.nn.GraphNorm(in_channels=hidden_size),
                              act="leaky_relu")
        self.head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.LeakyReLU(),
                                  layer_init(nn.Linear(hidden_size, n_actions), std=out_std))

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index)
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        x = self.head(x)
        return x


class GCNNetwork(nn.Module):

    def __init__(self, num_node_features: int, n_actions: int, hidden_size: int = 128, out_std=np.sqrt(2)):
        super(GCNNetwork, self).__init__()
        self.gnn = pyg.nn.GCN(in_channels=num_node_features, hidden_channels=hidden_size,
                              out_channels=hidden_size, num_layers=2)
        self.head = layer_init(nn.Linear(hidden_size, n_actions), std=out_std)

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index)
        x = pyg.nn.global_add_pool(x=x, batch=data.batch)
        x = self.head(x)
        return x


class SAGENetwork(nn.Module):
    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 3, hidden_size: int = 128,
                 out_std: float = np.sqrt(2)):
        super(SAGENetwork, self).__init__()
        self.gnn = pyg.nn.GraphSAGE(in_channels=num_node_features, hidden_channels=hidden_size,
                                    out_channels=hidden_size, num_layers=n_layers, act="leaky_relu")

        self.mem1 = pyg.nn.MemPooling(
            hidden_size, hidden_size, heads=4, num_clusters=10)
        self.mem2 = pyg.nn.MemPooling(
            hidden_size, n_actions, heads=4, num_clusters=1)

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index)
        x = F.leaky_relu(x)
        x, S1 = self.mem1(x, data.batch)
        x = F.leaky_relu(x)
        x, S2 = self.mem2(x)
        x = x.squeeze(1)
        return x


class GraphTransformerNetwork(nn.Module):
    def __init__(self, num_node_features: int, n_actions: int, n_layers: int = 2, hidden_size: int = 128):
        super(GraphTransformerNetwork, self).__init__()
        self.gnn = pyg.nn.GraphMultisetTransformer(in_channels=num_node_features, hidden_channels=hidden_size,
                                                   out_channels=n_actions,
                                                   num_heads=4,
                                                   layer_norm=True)

    def forward(self, data: Union[pyg.data.Data, pyg.data.Batch]):
        x = self.gnn(x=data.x, edge_index=data.edge_index, batch=data.batch)
        return x


class GAT(BasicGNN):
    r"""The Graph Neural Network from `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ or `"How Attentive are Graph Attention
    Networks?" <https://arxiv.org/abs/2105.14491>`_ papers, using the
    :class:`~torch_geometric.nn.GATConv` or
    :class:`~torch_geometric.nn.GATv2Conv` operator for message passing,
    respectively.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        num_layers (int): Number of message passing layers.
        out_channels (int, optional): If not set to :obj:`None`, will apply a
            final linear transformation to convert hidden node embeddings to
            output size :obj:`out_channels`. (default: :obj:`None`)
        v2 (bool, optional): If set to :obj:`True`, will make use of
            :class:`~torch_geometric.nn.conv.GATv2Conv` rather than
            :class:`~torch_geometric.nn.conv.GATConv`. (default: :obj:`False`)
        dropout (float, optional): Dropout probability. (default: :obj:`0.`)
        act (str or Callable, optional): The non-linear activation function to
            use. (default: :obj:`"relu"`)
        norm (torch.nn.Module, optional): The normalization operator to use.
            (default: :obj:`None`)
        jk (str, optional): The Jumping Knowledge mode
            (:obj:`"last"`, :obj:`"cat"`, :obj:`"max"`, :obj:`"lstm"`).
            (default: :obj:`"last"`)
        act_first (bool, optional): If set to :obj:`True`, activation is
            applied before normalization. (default: :obj:`False`)
        act_kwargs (Dict[str, Any], optional): Arguments passed to the
            respective activation function defined by :obj:`act`.
            (default: :obj:`None`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.GATConv` or
            :class:`torch_geometric.nn.conv.GATv2Conv`.
    """

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout, **kwargs)
