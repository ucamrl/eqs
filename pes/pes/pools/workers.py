from copy import deepcopy
from typing import Optional
from multiprocessing import Process

from pes.agents.policy_wrapper import PolicyWrapper


# Slave workers
class Worker(Process):

    def __init__(self,
                 name: str,
                 worker_idx: int,
                 pipe,
                 env_fn: callable,
                 policy: str,
                 gamma: float,
                 max_sim_step: Optional[int],
                 verbose: bool,
                 device="cpu"):
        super(Worker, self).__init__()
        """each worker holds a local env,
        env MUST `restore()` and `checkpoint()` env states"""
        self.name = name
        self.worker_idx = worker_idx
        self.pipe = pipe
        self.env_fn = env_fn
        self.verbose = verbose

        # read-only
        self.gamma = gamma
        self.policy = policy
        self.max_sim_step = max_sim_step
        self.device = device

        self.wrapped_env = None
        self.policy_wrapper = None

    # Initialize the environment
    def init_process(self):
        self.wrapped_env = self.env_fn()
        # TODO why is this needed? gym will know if `step()` is not called from outside??
        self.wrapped_env.reset()

    # Initialize the default policy
    def init_policy(self):
        self.policy_wrapper = PolicyWrapper(self.policy, "main", None,
                                            self.device)

    def run(self):
        """caller call start(), and this is the entry point"""
        self.init_process()
        self.init_policy()
        if self.verbose:
            print(f"[WORKER]> {self.name} Worker {self.worker_idx} Ready.")

        # example: https://github.com/openai/gym/blob/6a04d49722724677610e36c1f92908e72f51da0c/gym/vector/async_vector_env.py#L549
        try:
            while True:
                # Wait for tasks
                command, args = self._receive_safe_protocol()

                if command == "KillProc":
                    break

                elif command == "Expansion":
                    checkpoint_data, shallow_clone_node, saving_idx, task_idx = args

                    # Expansion
                    self.wrapped_env.restore(checkpoint_data)
                    expand_action = shallow_clone_node.select_expand_action()
                    next_state, reward, done, info = self.wrapped_env.step(
                        expand_action)
                    if done:
                        new_checkpoint_data = None
                    else:
                        new_checkpoint_data = self.wrapped_env.checkpoint()

                    child_saturated = False
                    if shallow_clone_node.is_head and info[
                            "stop_reason"] == "SATURATED":
                        child_saturated = True

                    # Send result back
                    item = (expand_action, next_state, reward, done,
                            child_saturated, new_checkpoint_data, saving_idx,
                            task_idx)
                    self._send_safe_protocol("ReturnExpansion", item)
                    # print(f" {self.worker_idx} Expansion OK")

                elif command == "Simulation":
                    # action_applied -> has not yet reach maximum depth
                    task_idx, checkpoint_data, action_applied, first_action = args

                    state = self.wrapped_env.restore(checkpoint_data)

                    if action_applied:
                        accu_reward = self._simulate(state)
                        self._send_safe_protocol("ReturnSimulation",
                                                 (task_idx, accu_reward))
                    else:
                        # When simulation invoked because of reaching maximum
                        # search depth, the expansion action has not applied
                        # Therefore, we need to execute it first anyway.
                        state, reward, done, _ = self.wrapped_env.step(
                            first_action)

                        accu_reward = reward
                        if not done:
                            accu_reward = self._simulate(state)
                        self._send_safe_protocol(
                            "ReturnSimulation",
                            (task_idx, accu_reward, reward, done))

                else:
                    raise RuntimeError(f"Received unknown command: {command}")

        except (KeyboardInterrupt, Exception) as e:
            print(f"[WORKER]> {self.name} Exception {e}")
        finally:
            # gracefully shut down
            # self.wrapped_env.close()
            print(f"[WORKER]> {self.name} Worker {self.worker_idx} Exit.")

    def _simulate(self, state):
        cnt = 0
        accu_reward = 0.0
        accu_gamma = 1.0
        start_state_value = 0.  # to tune?
        factor = 1  # to tune?
        # start_state_value = self.get_value(state)

        # NOTE <- if already done, then this simulation will not be scheduled
        done = False
        while not done:
            action_n = len(self.wrapped_env.get_action_space())
            action = self.policy_wrapper.get_action(state, action_n)

            state, reward, done, info = self.wrapped_env.step(action)

            # timeLimited truncate
            if self.max_sim_step is not None and cnt == self.max_sim_step and not done:
                done = True
                # get the final reward
                reward = self.wrapped_env.reward_func(
                    done, info, self.wrapped_env.egraph, self.wrapped_env.expr,
                    self.wrapped_env.base_cost)

            accu_reward += reward * accu_gamma
            accu_gamma *= self.gamma
            cnt += 1

        # if not done:
        #     accu_reward += self.get_value(state) * accu_gamma

        # Use V(s) to stabilize simulation return
        accu_reward = accu_reward * factor + start_state_value * (1.0 - factor)

        return accu_reward

    # Send message through pipe
    def _send_safe_protocol(self, command, args):
        success = False
        count = 0  # TODO count??
        while not success:
            self.pipe.send((command, args))

            ret = self.pipe.recv()
            if ret == command:
                if count >= 10:
                    print(f"[WORKER]> Worker {self.worker_idx} Send?")
                success = True

            count += 1

    # Receive message from pipe
    def _receive_safe_protocol(self):
        self.pipe.poll(None)
        command, args = self.pipe.recv()
        self.pipe.send(command)
        return deepcopy(command), deepcopy(args)
