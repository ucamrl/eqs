import numpy as np

import gym
from gym import spaces
from ..rejoice import *
from collections import OrderedDict, deque, namedtuple
from ..lib import Language
from typing import Tuple, Optional, Union
from ..graph_space import GraphSpace
import math

def remap(x, in_min, in_max, out_min, out_max):
  return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min

class EGraphEnv(gym.Env):
    """Custom gym env for the egraph rule selection task."""
    metadata = {'render.modes': []}

    def __init__(self, lang: Language, expr: any, node_limit: int = 10_000, use_shrink_action=True):
        super(EGraphEnv, self).__init__()
        self.step_count = 0
        self.lang = lang
        self.expr = expr
        self.orig_expr = expr
        self.rewrite_rules = lang.rewrite_rules()
        self.use_shrink_action = use_shrink_action
        self.global_step = 0

        num_actions = lang.num_rules + 1
        if use_shrink_action:
            num_actions += 1
        
        self.action_space = spaces.Discrete(num_actions)
        self.observation_space = GraphSpace(num_node_features=lang.num_node_features,
                                            dtype=np.int8,
                                            low=0,
                                            high=lang.get_feature_upper_bounds())
        self.reward_range = (-1, 1)
        self.node_limit = node_limit
        self.egraph, self.max_cost, self.prev_cost, self.best_seen_cost = None, None, None, None

        self.acc_rewrites = 0

    def set_global_step(self, global_step: int):
        self.global_step = global_step

    def step(self, action: any) -> Tuple[any, float, bool, dict]:
        self.step_count += 1
        info = {"actual_cost": self.prev_cost, "acc_rewrites": self.acc_rewrites}
        old_size = self.egraph.total_size()
        is_stop_action = action == self.lang.num_rules
        if is_stop_action:
            if self.step_count == 0:
                print("STOOPED AT first step")
            # Agent has chosen to stop optimizing and terminate current episode
            # punish agent if it ends the episode without any improvement
            # reward = -1.0 if self.prev_cost == self.max_cost else 0.0
            reward = -1.0 if self.step_count == 0 else 0.0
            # TODO: should the next obs be None or still the current state?
            # return self._get_obs(), reward, True, info
            return None, reward, True, info

        is_rebase_action = action == self.lang.num_rules + 1
        if is_rebase_action:
            # "rebase" the egraph to be the current extraction result.
            # this can shrink the egraph and help us recover from e-node explosion.
            # print("rebasing egraph")
            # print("prev_cost", self.prev_cost)
            old_size = self.egraph.total_size()
            best_cost, best_expr = self.egraph.extract(self.expr)
            best_cost = float(best_cost)
            # print("best_cost", best_cost, "best_expr", best_expr)
            self.expr = best_expr
            # re-create the egraph given this new expression
            self.egraph = EGraph()
            self.egraph.add(self.expr)
            self.prev_cost = float(self.egraph.extract(self.expr)[0])
            # print("rebase get_obs")
            new_obs = self._get_obs()
            # reward for rebase 
            new_size = self.egraph.total_size()
            if new_size == old_size:
                # Didn't get any e-graph reduction from rebasing, no point to this (at all)
                reward = -0.5
            else:
                # reward is always negative (we don't want to encourage this)
                # but bigger reduction to size == less negative reward
                # keep in mind that magnitudes of this MUST be smaller than pos reward from cost reduction
                # i.e. if a cost reduction can be achieved after a rebase, we want to encourage taking the rebase
                nodes_removed = old_size - new_size
                assert nodes_removed >= 0  # sanity check
                reward = -0.01
                # reward = -float(nodes_removed / self.node_limit)
            # print("rebase old size:", old_size, "new size:", new_size, "reward", reward)
            is_done = False
            info["actual_cost"] = self.prev_cost
            return new_obs, reward, is_done, info

        # Normal rewrite action choice path
        rewrite_to_apply = [self.rewrite_rules[action]]
        stop_reason, num_applications, *rest = self.egraph.run(
            rewrite_to_apply, iter_limit=1, node_limit=self.node_limit)
        self.acc_rewrites += num_applications
        info["acc_rewrites"] = self.acc_rewrites

        info["stop_reason"] = stop_reason
        if stop_reason == 'SATURATED':
            # if it was saturated, applying the rule did nothing; no need to re-extract
            reward = -0.2
        elif stop_reason == 'NODE_LIMIT' or stop_reason == 'TIME_LIMIT':
            reward = -1.0
        else:
            best_cost, best_expr = self.egraph.extract(self.expr)
            best_cost = float(best_cost)
            info["actual_cost"] = best_cost
            if best_cost == self.prev_cost:
                # (maybe) expanded egraph, but didn't get us a better extraction cost
                new_size = self.egraph.total_size()
                nodes_added = (new_size - old_size)
                assert nodes_added >= 0  # sanity check

                # more nodes added = more punishment!
                reward = -0.1 - float(nodes_added / self.node_limit)
            else:
                # give reward based upon improvement to extraction cost
                reward = 10 * (self.prev_cost - best_cost) / self.max_cost
                self.prev_cost = best_cost

        is_done = is_terminal(stop_reason)
        if stop_reason != "NODE_LIMIT":
            new_obs = self._get_obs()
            info["actions_available"] = sum(new_obs.action_mask)
        else:
            print("NODE LIMIT!!!")
            new_obs = None

        self.is_first_step = False
        return new_obs, reward, is_done, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[dict] = None,
    ) -> Union[any, "tuple[any, dict]"]:
        super().reset(seed=seed)
        self.step_count = 0
        self.egraph = EGraph()
        self.expr = self.orig_expr
        self.egraph.add(self.expr)
        self.acc_rewrites = 0
        self.max_cost = float(self.egraph.extract(self.expr)[0])
        self.prev_cost = self.max_cost
        self.best_seen_cost = self.max_cost
        # reward is normalized to (0, max_cost)
        # print("reset get_obs")
        new_obs = self._get_obs()
        info = {"actual_cost": self.prev_cost, "actions_available": sum(new_obs.action_mask)}
        if return_info:
            return new_obs, info
        else:
            return new_obs

    def _get_obs(self):
        return self.lang.encode_egraph(self.egraph, use_shrink_action=self.use_shrink_action, step=self.step_count)

    def close(self):
        pass


def is_terminal(stop_reason: str):
    """The episode should end if egg returns a STOP_REASON that indicates that the egraph has grown
        too large or extraaction is timing out."""
    if stop_reason == "ITERATION_LIMIT":
        # This means that the rewrite we took succeeded and nothing unexpected happened.
        return False
    elif stop_reason == "SATURATED":
        # Note that SATURATION isn't global saturation; it just means that the action we took didn't
        # change the egraph. This will happen frequently and is normal.
        return False
    else:
        return True
