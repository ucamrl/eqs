"""
Wrapper encapsulates Rust-level APIs
    MathLang & PropLang inheritance from there
"""
import sys
import functools
from typing import Protocol, NamedTuple
from collections import namedtuple

import numpy as np

from rust_lib import EGraph, Rewrite

# needed for safe expression generation
sys.setrecursionlimit(10**5)

# this is a class that can be used to instantiate
TestExprs = namedtuple("TestExprs", ["saturatable", "explodes"])


class ObjectView(object):

    def __init__(self, d):
        self.__dict__ = d


# A protocol type contains a set of typed methods and variables
# If an object has those methods and variables, it will match the protocol type
# similar to `trait` in Rust
class Language(Protocol):
    """A base Language for an equality saturation task.
    This will be passed to egg."""

    num_static_features = 4

    def get_supported_datatypes(self):
        # ["symbols", "integers"]
        return ["symbols"]

    @property
    def name(self):
        return type(self).__name__

    def op_tuple(self, op):
        """Convert an operator string (OpName, x, y, ...) into a named tuple"""
        name, *args = op
        # name becomes class name
        tup = NamedTuple(name, [(a, int) for a in args])
        # return the global symbol table
        # will not register in globals() if calling from other script
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
        # python rewrite rules -> Rust Rewrite
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
                        ([1] * self.num_operators) + ([1] * self.num_rules))

    @functools.cached_property
    def feature_names(self):
        features = ["is_eclass", "is_enode", "is_scalar", "is_terminal"]
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

    def gen_expr(self, root_op=None, p_leaf=0.6, depth=0, depth_limit=5):
        """Generate an arbitrary expression which abides by the language."""
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
                    op_children.append(
                        self.gen_expr(chosen_op, depth=depth + 1))
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

    def matches_to_lookup(self, eclass_ids: list[str], matches):
        # restructure the dict
        eclass_lookup = {k: [0] * self.num_rules for k in eclass_ids}

        for rule, ecids in matches.items():
            for ecid in ecids:
                eclass_lookup[ecid][self.rule_name_to_ind(rule)] = 1

        return eclass_lookup
