import gc
import time
import torch
import numpy as np
from typing import Optional

from pes.agents.uct_node import UCTnode
from pes.agents.policy_wrapper import PolicyWrapper
from pes.utils.checkpoint_manager import CheckpointManager


class MCTS_UCT:

    def __init__(
        self,
        env,
        budget: int,
        max_depth: Optional[int],
        max_width: Optional[int],
        max_sim_step: Optional[int],
        gamma: float,
        partial_cost_map: Optional[dict],  # None for now
        policy: str,
        device=torch.device("cpu")):

        self.budget = budget
        self.max_depth = max_depth
        self.max_width = max_width
        self.max_sim_step = max_sim_step
        self.partial_cost_map = partial_cost_map  # read-only
        self.gamma = gamma
        self.policy = policy
        self.device = device

        # Environment, must has `checkpoint()` and `restore()` methods
        self.wrapped_env = env

        # Checkpoint data manager
        self.checkpoint_data_manager = CheckpointManager()
        self.checkpoint_data_manager.hock_env("main", self.wrapped_env)

        # For MCTS tree
        self.root_node = None
        self.global_saving_idx = 0
        self.init_policy()

    def init_policy(self):
        self.policy_wrapper = PolicyWrapper(self.policy, "main",
                                            self.partial_cost_map, self.device)

    def start_planning(self, state, verbose=False) -> int:
        # skip planning if only one option
        action_n = len(self.wrapped_env.get_action_space())
        if action_n == 1:
            return 0

        # else start planning
        # Clear cache
        self.root_node = None
        self.global_saving_idx = 0
        self.checkpoint_data_manager.clear()
        gc.collect()

        # ckpt current state
        self.checkpoint_data_manager.checkpoint_env("main",
                                                    self.global_saving_idx)

        # Construct root node
        self.root_node = UCTnode(action_n=action_n,
                                 checkpoint_idx=self.global_saving_idx,
                                 parent=None,
                                 gamma=self.gamma,
                                 max_width=min(action_n, self.max_width)
                                 if self.max_width is not None else action_n,
                                 is_head=True)
        self.global_saving_idx += 1

        # sequential MCTS-UCT
        iter_cnt = []
        asts = []
        if verbose:
            print("[MCTS-UCT] starting planning...")
        for _ in range(self.budget):
            t0 = time.perf_counter()
            info = self.simulate_single_step()
            iter_cnt.append(info["count"])
            asts.append(info["ast_size"])
            t1 = time.perf_counter()
            if verbose:
                print(f"{t1-t0:.2f}s", end="; ")

        mean = np.mean(iter_cnt)
        std = np.std(iter_cnt)
        maxi = max(iter_cnt)
        mini = min(iter_cnt)
        # print(f"[MCTS-UCT] {cnt}/{self.budget} rollouts dones", end="; ")
        print(f"[MCTS-UCT] average iter count {mean:.2f} \u00B1 {std:.2f}",
              end="; ")
        print(f"Max {maxi}; Min {mini}", end=" - ")
        mean = np.mean(asts)
        std = np.std(asts)
        maxi = max(asts)
        mini = min(asts)
        print(f"average estimated ast size {mean:.2f} \u00B1 {std:.2f}",
              end="; ")
        print(f"Max {maxi}; Min {mini}")

        # UCT
        best_action = self.root_node.max_utility_action()

        # restore ckpt from the root
        self.checkpoint_data_manager.load_checkpoint_env(
            "main", self.root_node.checkpoint_idx)

        return best_action

    def simulate_single_step(self):
        # start from root node
        curr_node = self.root_node

        # Selection
        curr_depth = 1
        while True:
            if curr_node.no_child_available() or (
                    not curr_node.all_child_visited()
                    and curr_node != self.root_node and np.random.random() <
                    0.5) or (not curr_node.all_child_visited()
                             and curr_node == self.root_node):
                # If no child node has been expanded, we have to expand anyway.
                # Or if root node is not fully visited.
                # Or if non-root node is not fully visited and {with prob 1/2}.

                need_expansion = True
                break

            else:
                # compute child score via UCT
                action = curr_node.select_action()

            # just a tuple
            curr_node.update_history(action, curr_node.rewards[action])

            # env finishes or MCTS reaches depth limit
            if curr_node.dones[action] or (self.max_depth is not None
                                           and curr_depth >= self.max_depth):
                need_expansion = False
                break

            # else go to next level
            next_node = curr_node.children[action]
            curr_depth += 1
            curr_node = next_node

        # Expansion; make one-step transition
        if need_expansion:
            expand_action = curr_node.select_expand_action()

            # one-step step
            self.checkpoint_data_manager.load_checkpoint_env(
                "main", curr_node.checkpoint_idx)
            next_state, reward, done, info = self.wrapped_env.step(
                expand_action)
            # this ckpt belongs to curr_node's child
            self.checkpoint_data_manager.checkpoint_env(
                "main", self.global_saving_idx)

            # expansion starts from curr_node
            curr_node.rewards[expand_action] = reward
            curr_node.dones[expand_action] = done
            curr_node.update_history(action_taken=expand_action, reward=reward)
            next_action_space = len(self.wrapped_env.get_action_space())

            # if root node, add whether the child is saturated
            child_saturated = False
            if curr_node == self.root_node and info[
                    "stop_reason"] == "SATURATED":
                child_saturated = True

            curr_node.add_child(
                expand_action,
                next_action_space,
                min(self.max_width, next_action_space)
                if self.max_width is not None else next_action_space,
                self.global_saving_idx,
                prior_logits=self.policy_wrapper.get_prior_prob(
                    next_state, next_action_space),
                child_saturated=child_saturated)
            self.global_saving_idx += 1
        else:
            self.checkpoint_data_manager.load_checkpoint_env(
                "main", curr_node.checkpoint_idx)
            next_state, reward, done, info = self.wrapped_env.step(action)

            curr_node.rewards[action] = reward
            curr_node.dones[action] = done

        # Simulation
        # NOTE: rollout may never make it to the end, especially random policy
        # unless env guarantees time limited
        # done = False  # <--- NOTE if expension done then no rollout
        cnt = 0
        accu_reward = 0.0  # this is w.r.t the expanded node
        accu_gamma = 1.0
        while not done:

            action_n = len(self.wrapped_env.get_action_space())
            action = self.policy_wrapper.get_action(next_state, action_n)
            next_state, reward, done, info = self.wrapped_env.step(action)

            # timelimit truncate
            if self.max_sim_step is not None and cnt == self.max_sim_step and not done:
                done = True
                # get the final reward
                reward = self.wrapped_env.reward_func(
                    done, info, self.wrapped_env.egraph, self.wrapped_env.expr,
                    self.wrapped_env.base_cost)

            accu_reward += reward * accu_gamma
            accu_gamma *= self.gamma
            cnt += 1

        # BackUp; accu_reward is w.r.t the expanded node
        self.complete_update(curr_node, self.root_node, accu_reward)
        return info

    def close(self):
        pass

    @staticmethod
    def complete_update(curr_node, curr_node_head, accu_reward):
        # TODO impl object comparison, but default seems ok
        while curr_node != curr_node_head:
            accu_reward = curr_node.update(accu_reward)
            curr_node = curr_node.parent

        curr_node_head.update(accu_reward)
