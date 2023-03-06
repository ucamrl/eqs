import math
import random
import numpy as np

import torch
from torch.distributions.categorical import Categorical

from pes.utils.moving_averager import MovingAvegCalculator


class UCTnode:

    def __init__(self,
                 action_n: int,
                 checkpoint_idx: int,
                 parent,
                 gamma: float,
                 max_width: int,
                 prior_logits=None,
                 is_head=False,
                 allowed_actions=None):
        self.action_n = action_n
        # each node has a unique index;
        # the index measures a unique env state
        self.checkpoint_idx = checkpoint_idx
        self.parent = parent
        self.gamma = gamma
        self.max_width = max_width
        self.is_head = is_head
        self.allowed_actions = allowed_actions

        # node statistics
        self.children = [None for _ in range(self.action_n)]
        self.rewards = [0.0 for _ in range(self.action_n)]
        self.dones = [False for _ in range(self.action_n)]
        self.children_visit_count = [0 for _ in range(self.action_n)]
        self.Q_values = [0 for _ in range(self.action_n)]
        self.visit_count = 0
        if is_head:
            self.children_saturated = [False for _ in range(self.action_n)]

        # for expansion
        if prior_logits is not None:
            self.prior_logits = prior_logits
        else:
            self.prior_logits = np.ones([self.action_n],
                                        dtype=np.float32) / self.action_n

        # Record traverse history; store (a, r)
        self.traverse_history = None

        # Updated node count
        self.updated_node_count = 0

        # Moving average calculator; one per node
        self.moving_aveg_calculator = MovingAvegCalculator(window_length=500)

    def no_child_available(self) -> bool:
        # None children nodes are expanded.
        return self.updated_node_count == 0

    def all_child_visited(self) -> bool:
        # All child nodes have been visited and updated.
        # NOTE: if root, you want to explore ALL children
        # if not root, if OK to upperbound the action space
        if self.is_head:
            if self.allowed_actions is None:
                return self.updated_node_count == self.action_n
            else:
                return self.updated_node_count == len(self.allowed_actions)
        else:
            # NOTE: max_width is guaranteed to be
            # either upperbound or the num of action
            return self.updated_node_count == self.max_width

    def select_action(self) -> int:
        best_score = -float("inf")
        best_action = -1

        for action in range(self.action_n):
            if self.children[action] is None:
                continue

            if self.allowed_actions is not None and action not in self.allowed_actions:
                continue

            exploit_score = self.Q_values[action] / self.children_visit_count[
                action]
            explore_score = math.sqrt(1.0 * math.log(self.visit_count) /
                                      self.children_visit_count[action])
            score_std = self.moving_aveg_calculator.get_standard_deviation()
            score = exploit_score + score_std * explore_score

            # NOTE: if draw, then first encounter will be chosen
            if score > best_score:
                best_score = score
                best_action = action

        assert (best_action != -1), "best action == -1?"
        # if best_action == -1:
        #     print()
        #     print()
        #     best_action = 0
        return best_action

    def max_utility_action(self) -> int:
        assert (self.is_head), "only called by root node"
        best_score = -float("inf")
        best_action = -1

        for action in range(self.action_n):
            if self.children[action] is None:
                raise RuntimeError(
                    f"root {action} child is None; budget not enough?")

            if self.children_saturated[action]:
                # if this action causes saturation, just skip
                # see egg's explanation for saturation:
                # it means applying this rule adds nothing to the E-graph
                continue

            score = self.Q_values[action] / self.children_visit_count[action]

            # NOTE: if draw, then first encounter will be chosen
            if score > best_score:
                best_score = score
                best_action = action

        # assert (best_action != -1), "best action == -1??"
        if best_action == -1:
            print()
            print("[UCT-NODE][Warning] max_utility_action::", end=" ")
            print("best_action -1, all children are saturated")
            print()
            best_action = 0
        return best_action

    def select_expand_action(self):
        count = 0
        while True:
            if self.allowed_actions is None:
                if count < 20:
                    if torch.is_tensor(self.prior_logits):
                        logits = self.prior_logits
                    else:
                        logits = torch.Tensor(self.prior_logits)
                    dist = Categorical(logits=logits)
                    action = int(dist.sample())
                else:
                    action = np.random.randint(0, self.action_n)
            else:
                action = random.choice(self.allowed_actions)

            if count > 100:
                return action

            # try to expand s.t each child is selected at least once
            if self.children_visit_count[action] > 0 and count < 10:
                count += 1
                continue

            if self.children[action] is None:
                return action

            count += 1

        # simple expansion policy
        # NOTE: if need expansion, it means there are unexpanded children
        # for i in range(self.action_n):
        #     if self.children_visit_count[i] == 0:
        #         return i
        # raise RuntimeError(
        #     f"node {self.checkpoint_idx}; expansion all children visited")

    def update_history(self, action_taken, reward):
        self.traverse_history = (action_taken, reward)

    def update(self, accu_reward):
        assert self.traverse_history is not None, "None traverse_history"
        action_taken, reward = self.traverse_history
        accu_reward = reward + self.gamma * accu_reward

        if self.children_visit_count[action_taken] == 0:
            self.updated_node_count += 1

        self.children_visit_count[action_taken] += 1
        self.Q_values[action_taken] += accu_reward

        self.visit_count += 1

        self.moving_aveg_calculator.add_number(accu_reward)

        return accu_reward

    def add_child(self, action, action_n, max_width, checkpoint_idx,
                  prior_logits, child_saturated: bool):
        if child_saturated:
            assert (self.is_head)
            self.children_saturated[action] = True
        if self.children[action] is not None:
            # if use heuristic to expand,
            # this is possible
            node = self.children[action]
        else:
            node = UCTnode(action_n=action_n,
                           checkpoint_idx=checkpoint_idx,
                           parent=self,
                           gamma=self.gamma,
                           max_width=max_width,
                           prior_logits=prior_logits,
                           is_head=False)
            self.children[action] = node

        return node  # the returned child is not used

        # if self.children[action] is not None:
        #     raise RuntimeError(
        #         f"node {self.checkpoint_idx}; add_child {action} already exist"
        #     )
        # else:
        #     node = UCTnode(action_n=action_n,
        #                    checkpoint_idx=checkpoint_idx,
        #                    parent=self,
        #                    gamma=self.gamma,
        #                    max_width=max_width,
        #                    prior_logits=prior_logits,
        #                    is_head=False)
        #     self.children[action] = node
