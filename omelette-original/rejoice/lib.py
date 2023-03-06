import functools

from rejoice import *
from typing import Protocol, Union, NamedTuple
from collections import OrderedDict, namedtuple, Hashable
from rejoice.util import BytesIntEncoder
import torch
import torch_geometric as geom
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import torch_geometric.transforms as T
import time
import sys
import string

# needed for safe expression generation
sys.setrecursionlimit(10**5)

TestExprs = namedtuple("TestExprs", ["saturatable", "explodes"])


class ObjectView(object):
    def __init__(self, d):
        self.__dict__ = d


class Language(Protocol):
    """A base Language for an equality saturation task. This will be passed to egg."""

    num_static_features = 4

    def get_supported_datatypes(self):
        # ["symbols", "integers"]
        return ["symbols"]

    @property
    def name(self):
        return type(self).__name__

    def op_tuple(self, op):
        """Convert an operator string (name, x, y, ...) into a named tuple"""
        name, *args = op
        tup = NamedTuple(name, [(a, int) for a in args])
        globals()[name] = tup
        globals()[tup.__name__] = tup
        return tup

    def eclass_analysis(self, *args) -> any:
        ...

    def all_operators(self) -> list:
        ...

    def all_operators_obj(self):
        op_dict = self.all_operators_dict()
        return ObjectView(op_dict)

    def all_operators_dict(self):
        op_dict = dict([(operator.__name__.lower(), operator)
                       for operator in self.all_operators()])
        return op_dict

    def all_rules(self) -> "list[list]":
        ...

    def get_terminals(self) -> "list":
        ...
    
    def get_single_task_exprs(self) -> TestExprs:
        ...

    def get_multi_task_exprs(self) -> list:
        ...

    @functools.cached_property
    def num_terminals(self):
        return len(self.get_terminals())

    @functools.cached_property
    def num_operators(self):
        return len(self.all_operators())

    def rewrite_rules(self):
        rules = list()
        for rl in self.all_rules():
            name = rl[0]
            frm = rl[1]
            to = rl[2]
            rules.append(Rewrite(frm, to, name))
        return rules

    @property
    def num_node_features(self) -> int:
        return self.num_static_features + self.num_terminals + self.num_operators + self.num_rules

    def get_feature_upper_bounds(self):
        return np.array(([1] * self.num_static_features) +
                        ([1] * self.num_terminals) +
                        ([1] * self.num_operators) +
                        ([1] * self.num_rules))

    @functools.cached_property
    def feature_names(self):
        features = ["is_eclass",
                    "is_enode",
                    "is_scalar",
                    "is_terminal"]
        terminal_names = [str(t) for t in self.get_terminals()]
        op_names = [op.__name__ for op in self.all_operators()]
        rule_names = [rule[0] for rule in self.all_rules()]
        return features + terminal_names + op_names + rule_names

    @functools.cached_property
    def op_to_ind(self):
        op_to_ind_table = {}
        for ind, op in enumerate(self.all_operators()):
            op_to_ind_table[op] = ind
        return op_to_ind_table

    def gen_expr(self, root_op=None, p_leaf=0.6, depth=0):
        """Generate an arbitrary expression which abides by the language."""
        depth_limit = 6
        ops = self.all_operators()
        root = np.random.choice(ops) if root_op is None else root_op
        children = []
        for i in range(len(root._fields)):
            if np.random.uniform(0, 1) < p_leaf or depth >= depth_limit:
                if np.random.uniform(0, 1) < 0.5:
                    children.append(np.random.choice(self.get_terminals()))
                else:
                    if "symbols" in self.get_supported_datatypes():
                        symbols = ["a", "b", "c", "d"]
                        # symbols = list(string.ascii_lowercase)
                        children.append(np.random.choice(symbols))
                    if "integers" in self.get_supported_datatypes():
                        children.append(np.random.randint(0, 5)) 
            else:
                chosen_op = np.random.choice(ops)
                op_children = []
                for j in range(len(chosen_op._fields)):
                    op_children.append(self.gen_expr(chosen_op, depth=depth+1))
                children.append(chosen_op(*op_children))
        return root(*children)


    def operator_names(self):
        return [op.__name__.lower() for op in self.all_operators()]
    
    @functools.cached_property
    def num_rules(self):
        return len(self.all_rules())

    def rule_name_to_ind(self, rname: str) -> int:
        rl_names = [rl[0] for rl in self.all_rules()]
        return rl_names.index(rname)

    @functools.cached_property
    def rule_names(self) -> list[str]:
        rl_names = [rl[0] for rl in self.all_rules()]
        return rl_names

    def matches_to_lookup(self, eclass_ids: "list[str]", matches):
        # restructure the dict
        eclass_lookup = {k: [0]*self.num_rules for k in eclass_ids}

        for rule, ecids in matches.items():
            for ecid in ecids:
                eclass_lookup[ecid][self.rule_name_to_ind(rule)] = 1

        return eclass_lookup

    def encode_egraph(self, egraph: EGraph, y=None, use_shrink_action=False, step=None) -> geom.data.Data:
        egraph.rebuild()
        # first_stamp = int(round(time.time() * 1000))
        num_enodes = egraph.num_enodes()
        eclass_ids = egraph.eclass_ids()
        num_eclasses = len(eclass_ids)
        eclass_enode_edges = torch.zeros([2, num_enodes])
        eclass_enode_edge_attr = torch.tensor(
            [1, 0]).expand(eclass_enode_edges.size()[-1], -1)

        x = torch.zeros([num_eclasses + num_enodes, self.num_node_features])
        x[:num_eclasses, 0] = 1  # make eclass nodes
        x[num_eclasses:, 1] = 1  # mark enodes

        curr = num_eclasses
        edge_curr = 0

        eclass_to_ind = dict(zip(eclass_ids, range(num_eclasses)))
        classes = egraph.classes()

        all_node_edges = []

        term_start = self.num_static_features
        op_start = term_start + self.num_terminals
        rule_start = op_start + self.num_operators

        matches = egraph.match_rules(self.rewrite_rules())
        eclass_to_rule_inds = self.matches_to_lookup(eclass_ids, matches)

        for eclass_id, (data, nodes) in classes.items():
            eclass_ind = eclass_to_ind[eclass_id]

            x[eclass_ind][rule_start:] = torch.Tensor(
                eclass_to_rule_inds[eclass_id])

            num_eclass_nodes = len(nodes)
            # create edges from eclass to member enodes
            eclass_enode_edges[0, edge_curr:(
                edge_curr + num_eclass_nodes)] = eclass_ind
            eclass_enode_edges[1, edge_curr:(
                edge_curr + num_eclass_nodes)] = torch.arange(curr, curr + num_eclass_nodes)
            edge_curr = edge_curr + num_eclass_nodes

            for node in nodes:
                # we only want to encode if they're terminals... everything else will cause learning confusion.
                if isinstance(node, int) or isinstance(node, float) or isinstance(node, bool) or isinstance(node, str) or isinstance(node, np.bool_) or isinstance(node, np.int64):
                    try:
                        term_ind = self.get_terminals().index(node)
                        x[curr, 3] = 1
                        x[curr, term_start + term_ind] = 1
                    except ValueError:
                        # it's an unknown scalar (not in terminals list)
                        x[curr, 2] = 1
                else:
                    # encode operator type
                    x[curr, op_start + self.op_to_ind[type(node)]] = 1
                    # connect to child eclasses
                    if isinstance(node, tuple):
                        all_node_edges.append(torch.stack([torch.full([len(node)], curr),
                                              torch.Tensor([eclass_to_ind[str(ecid)] for ecid in node])]))
                curr += 1

        edge_index = torch.concat(
            [eclass_enode_edges, *all_node_edges], dim=1).long()


        enode_eclass_edge_attr = torch.Tensor(
            [0, 1]).expand(torch.concat(all_node_edges, dim=1).size()[-1], -1) if len(all_node_edges) >0 else torch.Tensor([])

        edge_attr = torch.concat(
            [eclass_enode_edge_attr, enode_eclass_edge_attr])

        edge_index, edge_attr = geom.utils.add_remaining_self_loops(
            edge_index, edge_attr, fill_value=0.)

        action_mask = x[:, rule_start:].sum(dim=0).clamp(0, 1)
        if use_shrink_action:
            action_mask = torch.cat((action_mask, torch.ones(2)))
            if step < 100:
                 action_mask[-2] = 0
        else:
            action_mask = torch.cat((action_mask, torch.ones(1)))
            # if step < 100:
            #     action_mask[-1] = 0

        if y is not None:
            y = torch.Tensor([y]).long()

        data = geom.data.Data(x=x, edge_index=edge_index, edge_attr=edge_attr,
                              y=y, action_mask=action_mask)
        # second_stamp = int(round(time.time() * 1000))
        # Calculate the time taken in milliseconds
        # time_taken = second_stamp - first_stamp
        # print("time_taken", time_taken, data)
        return data

    def decode_node(self, node: torch.Tensor):
        term_start = self.num_static_features
        op_start = term_start + self.num_terminals
        rule_start = op_start + self.num_operators

        is_eclass = bool(node[0])
        is_operator = bool(node[1])
        is_scalar = bool(node[2])
        is_terminal = bool(node[3])

        node_data = {}
        if is_eclass:
            node_data["name"] = "eclass"
            node_data["value"] = [f for ind, f in enumerate(
                self.feature_names[rule_start:]) if node[rule_start + ind] == 1]
        elif is_terminal:
            node_data["name"] = "terminal"
            node_data["value"] = [f for ind, f in enumerate(
                self.feature_names[term_start:op_start]) if node[term_start + ind] == 1]
        elif is_scalar:
            node_data["name"] = "scalar"
            node_data["value"] = "?"
        elif is_operator:
            node_data["name"] = "operator"
            node_data["value"] = [f for ind, f in enumerate(
                self.feature_names[op_start:rule_start]) if node[op_start + ind] == 1]

        return node_data

    def eval_expr(self, expr_str: str):
        ops = self.all_operators_dict() # needed to avoid pickling errors
        return eval(expr_str)

    def viz_egraph(self, data):
        """Vizualize a PyTorch Geometric data object containing an egraph."""
        print("vizualizing egraph", data)
        g = geom.utils.to_networkx(data, node_attrs=['x'])

        for u, data in g.nodes(data=True):
            decoded = self.decode_node(data["x"])
            data["name"] = decoded["name"]
            data["value"] = decoded["value"]
            del data['x']

        node_labels = {}
        for u, data in g.nodes(data=True):
            node_labels[u] = data['name'] + str(data["value"])

        pos = nx.nx_agraph.graphviz_layout(g, prog="dot")
        nx.draw(g, labels=node_labels, pos=pos)
        plt.savefig("./test_eg.png")
        return g
