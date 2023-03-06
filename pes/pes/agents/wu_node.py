import numpy as np
from copy import deepcopy
import math

from pes.utils.moving_averager import MovingAvegCalculator


class WU_UCTnode:

    def __init__(self,
                 action_n,
                 checkpoint_idx,
                 parent,
                 gamma,
                 max_width,
                 prior_prob=None,
                 is_head=False):
        self.action_n = action_n
        self.checkpoint_idx = checkpoint_idx
        self.parent = parent
        self.gamma = gamma
        self.is_head = is_head
        self.max_width = max_width

        self.children = [None for _ in range(self.action_n)]
        self.rewards = [0.0 for _ in range(self.action_n)]
        self.dones = [False for _ in range(self.action_n)]
        self.children_visit_count = [0 for _ in range(self.action_n)]
        self.children_completed_visit_count = [0 for _ in range(self.action_n)]
        self.Q_values = [0 for _ in range(self.action_n)]
        self.visit_count = 0
        if is_head:
            self.children_saturated = [False for _ in range(self.action_n)]

        if prior_prob is not None:
            raise RuntimeError("prior_prob should be None")
        else:
            self.prior_prob = np.ones([self.action_n],
                                      dtype=np.float32) / self.action_n

        # Record traverse history
        self.traverse_history = dict()

        # Visited node count
        self.visited_node_count = 0

        # Updated node count
        self.updated_node_count = 0

        # Moving average calculator
        self.moving_aveg_calculator = MovingAvegCalculator(window_length=500)

    def no_child_available(self):
        # All child nodes have not been expanded.
        return self.updated_node_count == 0

    def all_child_visited(self):
        # All child nodes have been visited (not necessarily updated).
        if self.is_head or self.max_width is None:
            return self.visited_node_count == self.action_n
        else:
            return self.visited_node_count == self.max_width

    def all_child_updated(self):
        # All child nodes have been updated.
        if self.is_head or self.max_width is None:
            return self.updated_node_count == self.action_n
        else:
            return self.updated_node_count == self.max_width

    # Shallowly clone itself, contains necessary data only.
    # it will be sent over to worker for expansion tasks
    def shallow_clone(self):
        node = WU_UCTnode(action_n=self.action_n,
                          checkpoint_idx=self.checkpoint_idx,
                          parent=None,
                          gamma=self.gamma,
                          max_width=self.max_width,
                          prior_prob=None,
                          is_head=self.is_head)

        for action in range(self.action_n):
            if self.children[action] is not None:
                node.children[action] = 1

        node.children_visit_count = deepcopy(self.children_visit_count)
        node.children_completed_visit_count = deepcopy(
            self.children_completed_visit_count)

        node.visited_node_count = self.visited_node_count
        node.updated_node_count = self.updated_node_count

        node.action_n = self.action_n
        node.max_width = self.max_width

        node.prior_prob = self.prior_prob.copy()

        return node

    # Select action according to the P-UCT tree policy
    def select_action(self):
        best_score = -float("inf")
        best_action = -1

        for action in range(self.action_n):
            if self.children[action] is None:
                continue

            exploit_score = self.Q_values[
                action] / self.children_completed_visit_count[action]
            explore_score = math.sqrt(2.0 * math.log(self.visit_count) /
                                      self.children_visit_count[action])
            score_std = self.moving_aveg_calculator.get_standard_deviation()
            score = exploit_score + score_std * 2.0 * explore_score

            if score > best_score:
                best_score = score
                best_action = action

        assert (best_action != -1)
        return best_action

    # Return the action with maximum utility.
    def max_utility_action(self):
        best_score = -float("inf")
        best_action = -1

        for action in range(self.action_n):
            if self.children_saturated[action]:
                continue
            if self.children[action] is None:
                continue

            score = self.Q_values[
                action] / self.children_completed_visit_count[action]

            if score > best_score:
                best_score = score
                best_action = action

        assert (best_action != -1)
        return best_action

    # Choose an action to expand
    def select_expand_action(self):
        count = 0
        while True:
            if count < 20:
                action = self.categorical(self.prior_prob)
            else:
                action = np.random.randint(0, self.action_n)

            if count > 100:
                return action

            if self.children_visit_count[action] > 0 and count < 10:
                count += 1
                continue

            if self.children[action] is None:
                return action

            count += 1

    # Update traverse history, used to perform update
    def update_history(self, idx, action_taken, reward):
        if idx in self.traverse_history:
            return False
        else:
            self.traverse_history[idx] = (action_taken, reward)
            return True

    # Incomplete update, called by WU_UCT.py
    def update_incomplete(self, idx):
        action_taken = self.traverse_history[idx][0]

        if self.children_visit_count[action_taken] == 0:
            self.visited_node_count += 1

        self.children_visit_count[action_taken] += 1
        self.visit_count += 1

    # Complete update, called by WU_UCT.py
    def update_complete(self, idx, accu_reward):
        if idx not in self.traverse_history:
            raise RuntimeError(
                "idx {} should be in traverse_history".format(idx))
        else:
            item = self.traverse_history.pop(idx)
            action_taken = item[0]
            reward = item[1]

        accu_reward = reward + self.gamma * accu_reward

        if self.children_completed_visit_count[action_taken] == 0:
            self.updated_node_count += 1

        self.children_completed_visit_count[action_taken] += 1
        self.Q_values[action_taken] += accu_reward

        self.moving_aveg_calculator.add_number(accu_reward)

        return accu_reward

    # Add a child to current node.
    def add_child(self, action, checkpoint_idx, gamma, max_width, prior_prob,
                  child_saturated: bool):
        assert (isinstance(checkpoint_idx, int)), f"{type(checkpoint_idx)}?"
        if child_saturated:
            assert (self.is_head)
            self.children_saturated[action] = True

        if self.children[action] is not None:
            node = self.children[action]
        else:
            # NOTE: assume fixed action space now
            node = WU_UCTnode(action_n=self.action_n,
                              checkpoint_idx=checkpoint_idx,
                              parent=self,
                              gamma=gamma,
                              max_width=max_width,
                              prior_prob=prior_prob)

            self.children[action] = node

        return node

    # Draw a sample from the categorical distribution parametrized by 'pvals'.
    @staticmethod
    def categorical(pvals):
        num = np.random.random()
        for i in range(pvals.size):
            if num < pvals[i]:
                return i
            else:
                num -= pvals[i]

        return pvals.size - 1
