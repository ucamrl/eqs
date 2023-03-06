import gc
import time
import numpy as np
from copy import deepcopy
from typing import Optional
from collections import namedtuple

from pes.agents.wu_node import WU_UCTnode
from pes.pools.pool_manager import PoolManager
from pes.utils.checkpoint_manager import CheckpointManager

EXP_TASK = namedtuple("expansion_task",
                      ["ckpt", "shallow_copy_node", "curr_node"])
SIM_TASK = namedtuple(
    "simulation_task",
    ["action", "curr_node", "saving_idx", "action_applied", "child_saturated"])


class WU_UCT:

    def __init__(
            self,
            # thunk for make envs
            env,  # main env, also used outside
            exp_env_fns: list[callable],
            sim_env_fns: list[callable],
            # wu-uct params
            budget: int,
            max_depth: Optional[int],
            max_width: Optional[int],
            max_sim_step: Optional[int],
            gamma: float,
            expansion_worker_num: int,
            simulation_worker_num: int,
            policy="Random",
            device="cpu"):
        self.budget = budget
        self.max_depth = max_depth
        self.max_width = max_width
        self.gamma = gamma

        # Environment
        self.wrapped_env = env

        # Expansion worker pool
        self.expansion_worker_pool = PoolManager(
            name="Expansion",
            worker_num=expansion_worker_num,
            env_fns=exp_env_fns,
            policy=policy,
            gamma=gamma,
            max_sim_step=max_sim_step,
            device=device)

        # Simulation worker pool
        self.simulation_worker_pool = PoolManager(
            name="Simulation",
            worker_num=simulation_worker_num,
            env_fns=sim_env_fns,
            policy=policy,
            gamma=gamma,
            max_sim_step=max_sim_step,
            device=device)

        # Checkpoint data manager
        self.checkpoint_data_manager = CheckpointManager()
        self.checkpoint_data_manager.hock_env("main", self.wrapped_env)

        # For MCTS tree
        self.root_node = None
        self.global_saving_idx = 0

        # Task recorder;
        # dict; task index -> data
        self.expansion_tasks = {}
        self.simulation_tasks = {}
        # list; list of task index
        self.pending_expansion_tasks = []
        self.pending_simulation_tasks = []

        # t_complete in the origin paper,
        # measures the completed number of simulations
        self.simulation_count = 0

    # This is the planning process of P-UCT.
    # Starts from a tree with a root node only,
    # P-UCT performs selection, expansion, simulation,
    # and backpropagation on it.
    def start_planning(self, state, verbose=False):
        # skip planning if only one option
        action_n = len(self.wrapped_env.get_action_space())
        if action_n == 1:
            return 0

        # Clear cache
        self.root_node = None
        self.global_saving_idx = 0
        self.checkpoint_data_manager.clear()

        self.expansion_tasks.clear()
        self.pending_expansion_tasks.clear()
        self.simulation_tasks.clear()
        self.pending_simulation_tasks.clear()

        gc.collect()

        # Free all workers
        self.expansion_worker_pool.wait_until_all_envs_idle()
        self.simulation_worker_pool.wait_until_all_envs_idle()

        # Construct root node
        self.checkpoint_data_manager.checkpoint_env("main",
                                                    self.global_saving_idx)
        self.root_node = WU_UCTnode(action_n=action_n,
                                    checkpoint_idx=self.global_saving_idx,
                                    parent=None,
                                    gamma=self.gamma,
                                    max_width=self.max_width,
                                    is_head=True)

        # An index used to retrieve game-states
        self.global_saving_idx += 1

        # t_complete in the origin paper,
        # measures the completed number of simulations
        self.simulation_count = 0

        # Repeatedly invoke the master loop (Figure 2 of the paper)
        for sim_idx in range(self.budget):
            # NOTE: this is the master process, so returns immediately
            self._simulate_single_step(sim_idx)
            # print(f"[WU-UCT] {sim_idx}")

        # there are always incomplete simulation work (some are stragglers)
        # that is not used to update the statistics
        print(
            f"[WU-UCT] complete count: {self.simulation_count}/{self.budget}")

        time.sleep(3)
        t0 = time.perf_counter()
        self.expansion_worker_pool.kill_straggler()
        t1 = time.perf_counter()
        self.simulation_worker_pool.kill_straggler()
        t2 = time.perf_counter()
        print(f"[WU-UCT] Expansion straggler {t1-t0:.2f}s", end="; ")
        print(f"Simulation straggler {t2-t1:.2f}s")

        # Select the best root action
        best_action = self.root_node.max_utility_action()

        # Retrieve the game-state before simulation begins
        self.checkpoint_data_manager.load_checkpoint_env(
            "main", self.root_node.checkpoint_idx)

        return best_action

    def _simulate_single_step(self, sim_idx: int):
        # Go into root node
        curr_node = self.root_node

        # Selection
        curr_depth = 1
        while True:
            if curr_node.no_child_available() or (
                    not curr_node.all_child_visited()
                    and curr_node != self.root_node and np.random.random() <
                    0.5) or (not curr_node.all_child_visited()
                             and curr_node == self.root_node):
                # If no child node has been updated, we have to expand anyway.
                # Or if root node is not fully visited.
                # Or if non-root node is not fully visited and {with prob 1/2}.

                cloned_curr_node = curr_node.shallow_clone()
                checkpoint_data = self.checkpoint_data_manager.retrieve(
                    curr_node.checkpoint_idx)

                # Record the task
                self.expansion_tasks[sim_idx] = EXP_TASK(
                    ckpt=checkpoint_data,
                    shallow_copy_node=cloned_curr_node,
                    curr_node=curr_node)
                self.pending_expansion_tasks.append(sim_idx)

                need_expansion = True
                break

            else:
                action = curr_node.select_action()

            curr_node.update_history(sim_idx, action,
                                     curr_node.rewards[action])

            if curr_node.dones[action] or (self.max_depth is not None
                                           and curr_depth >= self.max_depth):
                # Exceed maximum depth
                need_expansion = False
                break

            # XXX what is this???
            if curr_node.children[action] is None:
                raise RuntimeError(
                    f"[WU-UCT] IMPOSSIBLE {curr_depth};; {action}")
                need_expansion = False
                break

            # one-level deeper
            next_node = curr_node.children[action]
            curr_depth += 1
            curr_node = next_node

        # Expansion
        if not need_expansion:
            if not curr_node.dones[action]:
                print(
                    "[WU-UCT][Warning] reach maximum depth {curr_depth}/{self.max_depth}"
                )
                raise RuntimeError("maximum depth is not yet supported")
                # Reach maximum depth but have not terminate.
                # Record simulation task.

                self.simulation_tasks[sim_idx] = SIM_TASK(
                    action=action,
                    curr_node=curr_node,
                    saving_idx=curr_node.checkpoint_idx,
                    action_applied=False,
                    child_saturated=False)
                self.pending_simulation_tasks.append(sim_idx)
            else:
                # Reach terminal node.
                # In this case, update directly.

                self.incomplete_update(curr_node, self.root_node, sim_idx)
                # TODO update with 0 accu_reward??
                self.complete_update(curr_node, self.root_node, 0.0, sim_idx)

                self.simulation_count += 1

        else:
            # Assign tasks to idle server (schedule loop)
            while len(self.pending_expansion_tasks
                      ) > 0 and self.expansion_worker_pool.has_idle_server():
                # Get a task
                curr_idx = np.random.randint(0,
                                             len(self.pending_expansion_tasks))
                task_idx = self.pending_expansion_tasks.pop(curr_idx)

                # Assign the task to server; only send the shallow copy!
                checkpoint_data, cloned_curr_node, _ = self.expansion_tasks[
                    task_idx]
                self.expansion_worker_pool.assign_expansion_task(
                    checkpoint_data, cloned_curr_node, self.global_saving_idx,
                    task_idx)
                self.global_saving_idx += 1
                # print(f"expansion scheduled {self.global_saving_idx-1}")

            # if high occupancy, wait for an expansion task to complete
            if self.expansion_worker_pool.server_occupied_rate() >= 0.99:
                expand_action, next_state, reward, done, child_saturated, checkpoint_data, saving_idx, task_idx = self.expansion_worker_pool.get_complete_expansion_task(
                )

                # NOTE: this is the `true` self
                # the shallow copy is only used for expansion
                _, _, curr_node = self.expansion_tasks.pop(task_idx)
                curr_node.update_history(task_idx, expand_action, reward)

                # Record info; expand_action is w.r.t curr_node
                curr_node.dones[expand_action] = done
                curr_node.rewards[expand_action] = reward

                if done:
                    # If this expansion result in a terminal node,
                    # perform update directly (simulation is not needed)

                    # else add_child will be done after simulation!
                    curr_node.add_child(expand_action,
                                        saving_idx,
                                        self.gamma,
                                        self.max_width,
                                        prior_prob=None,
                                        child_saturated=child_saturated)

                    assert (checkpoint_data is None)
                    self.incomplete_update(curr_node, self.root_node, task_idx)
                    # TODO update with 0 accu_reward??
                    self.complete_update(curr_node, self.root_node, 0.0,
                                         task_idx)

                    self.simulation_count += 1

                else:
                    # Schedule the task to the simulation task buffer.

                    assert (checkpoint_data is not None)
                    self.checkpoint_data_manager.store(saving_idx,
                                                       checkpoint_data)

                    self.simulation_tasks[task_idx] = SIM_TASK(
                        action=expand_action,
                        curr_node=curr_node,
                        saving_idx=saving_idx,
                        action_applied=True,
                        child_saturated=child_saturated)
                    self.pending_simulation_tasks.append(task_idx)
                # print(f"expansion done {saving_idx}")

        # Assign simulation tasks to idle environment server
        while len(self.pending_simulation_tasks
                  ) > 0 and self.simulation_worker_pool.has_idle_server():
            # Get a task
            idx = np.random.randint(0, len(self.pending_simulation_tasks))
            task_idx = self.pending_simulation_tasks.pop(idx)

            # NOTE: this_action_applied denotes whether this_action
            # has been applied or not
            this_action, this_node, this_saving_idx, this_action_applied, _ = self.simulation_tasks[
                task_idx]
            this_checkpoint_data = self.checkpoint_data_manager.retrieve(
                this_saving_idx)

            # Assign the task to server
            self.simulation_worker_pool.assign_simulation_task(
                task_idx, this_checkpoint_data, this_action_applied,
                this_action)

            # Perform incomplete update
            self.incomplete_update(
                this_node,  # This is the corresponding node
                self.root_node,
                task_idx)

        # Wait for a simulation task to complete
        if self.simulation_worker_pool.server_occupied_rate() >= 0.99:
            args = self.simulation_worker_pool.get_complete_simulation_task()
            if len(args) == 2:
                task_idx, accu_reward = args
            else:
                raise RuntimeError("maximum depth is not yet supported")
                task_idx, accu_reward, reward, done = args
            expand_action, curr_node, saving_idx, action_applied, child_saturated = self.simulation_tasks.pop(
                task_idx)

            if action_applied:
                # if action_applied, imply not yet reach maximum depth
                curr_node.add_child(expand_action,
                                    saving_idx,
                                    self.gamma,
                                    self.max_width,
                                    prior_prob=None,
                                    child_saturated=child_saturated)
            else:
                raise RuntimeError("maximum depth is not yet supported")
                assert (len(args) == 4), f"{len(args)}??"
                curr_node.rewards[expand_action] = reward
                curr_node.dones[expand_action] = done

            # Complete Update
            self.complete_update(curr_node, self.root_node, accu_reward,
                                 task_idx)

            self.simulation_count += 1

    def close(self):
        # Free sub-processes
        self.expansion_worker_pool.close_pool()
        self.simulation_worker_pool.close_pool()

    # Incomplete update allows to track unobserved samples
    # (Algorithm 2 in the paper)
    @staticmethod
    def incomplete_update(curr_node, curr_node_head, idx):
        while curr_node != curr_node_head:
            curr_node.update_incomplete(idx)
            curr_node = curr_node.parent

        curr_node_head.update_incomplete(idx)

    # Complete update tracks the observed samples (Algorithm 3 in the paper)
    @staticmethod
    def complete_update(curr_node, curr_node_head, accu_reward, idx):
        while curr_node != curr_node_head:
            accu_reward = curr_node.update_complete(idx, accu_reward)
            curr_node = curr_node.parent

        curr_node_head.update_complete(idx, accu_reward)
