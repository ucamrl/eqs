import time
from multiprocessing import Pipe
from copy import deepcopy
from typing import Optional

import torch

from pes.pools.workers import Worker


# Works in the main process and manages sub-workers
class PoolManager:

    def __init__(self,
                 name,
                 worker_num,
                 env_fns,
                 policy: str,
                 gamma: float,
                 max_sim_step: Optional[int],
                 device="cpu"):
        self.name = name
        self.env_fns = env_fns
        self.policy = policy
        self.gamma = gamma
        self.max_sim_step = max_sim_step
        self.worker_num = worker_num

        # Buffer for workers and pipes
        self.workers = []
        self.pipes = []
        # Worker status: 0 for idle, 1 for busy
        self.worker_status = [0 for _ in range(worker_num)]

        # CUDA device parallelization; ignore for now
        # if multiple cuda devices exist, use them all
        if torch.cuda.is_available():
            torch_device_num = torch.cuda.device_count()
        else:
            torch_device_num = 0
        assert (device == "cpu"), "not support gpu now"

        # Initialize workers
        for worker_idx in range(worker_num):
            parent_pipe, child_pipe = Pipe()
            self.pipes.append(parent_pipe)

            worker = Worker(
                name=name,
                worker_idx=worker_idx,
                pipe=child_pipe,
                env_fn=env_fns[worker_idx],
                policy=policy,
                gamma=gamma,
                max_sim_step=max_sim_step,
                verbose=True,
                device=device + ":" +
                str(int(torch_device_num * worker_idx / worker_num))
                if device == "cuda" else device)
            # https://github.com/openai/gym/blob/6a04d49722724677610e36c1f92908e72f51da0c/gym/vector/async_vector_env.py#L83
            worker.daemon = True
            self.workers.append(worker)

        # Start workers
        for worker in self.workers:
            worker.start()

    def has_idle_server(self):
        for status in self.worker_status:
            if status == 0:
                return True
        return False

    def server_occupied_rate(self):
        occupied_count = 0.0

        for status in self.worker_status:
            occupied_count += status

        return occupied_count / self.worker_num

    def _find_idle_worker(self):
        for idx, status in enumerate(self.worker_status):
            if status == 0:
                self.worker_status[idx] = 1
                return idx
        assert (False), "no idle worker?"

    def assign_expansion_task(self, checkpoint_data, curr_node,
                              global_saving_idx, task_simulation_idx):
        worker_idx = self._find_idle_worker()

        self._send_safe_protocol(worker_idx, "Expansion",
                                 (checkpoint_data, curr_node,
                                  global_saving_idx, task_simulation_idx))

        self.worker_status[worker_idx] = 1

    def assign_simulation_task(self, task_idx, checkpoint_data,
                               action_applied: bool, first_action: int):
        worker_idx = self._find_idle_worker()

        self._send_safe_protocol(
            worker_idx, "Simulation",
            (task_idx, checkpoint_data, action_applied, first_action))

        self.worker_status[worker_idx] = 1

    def get_complete_expansion_task(self):
        selected_worker_idx = -1
        ok = False
        while not ok:
            for worker_idx in range(self.worker_num):
                item = self._receive_safe_protocol_tapcheck(worker_idx)
                if item is not None:
                    selected_worker_idx = worker_idx
                    ok = True
                    break

        command, args = item
        assert command == "ReturnExpansion"
        assert selected_worker_idx != -1

        # Set to idle
        self.worker_status[selected_worker_idx] = 0

        return args

    def get_complete_simulation_task(self):
        selected_worker_idx = -1
        ok = False
        while not ok:
            for worker_idx in range(self.worker_num):
                item = self._receive_safe_protocol_tapcheck(worker_idx)
                if item is not None:
                    selected_worker_idx = worker_idx
                    ok = True
                    break

        command, args = item
        assert command == "ReturnSimulation"
        assert selected_worker_idx != -1

        # Set to idle
        self.worker_status[selected_worker_idx] = 0

        return args

    def _send_safe_protocol(self, worker_idx, command, args):
        success = False
        while not success:
            self.pipes[worker_idx].send((command, args))

            ret = self.pipes[worker_idx].recv()
            if ret == command:
                success = True

    def _receive_safe_protocol(self, worker_idx):
        # Return whether there is any data available to be read.
        self.pipes[worker_idx].poll(None)  # block until readable

        command, args = self.pipes[worker_idx].recv()
        self.pipes[worker_idx].send(command)

        return deepcopy(command), deepcopy(args)

    def _receive_safe_protocol_tapcheck(self, worker_idx):
        flag = self.pipes[worker_idx].poll()
        # time.sleep(1)
        if not flag:
            return None

        command, args = self.pipes[worker_idx].recv()

        self.pipes[worker_idx].send(command)

        return deepcopy(command), deepcopy(args)

    def wait_until_all_envs_idle(self):
        for worker_idx in range(self.worker_num):
            if self.worker_status[worker_idx] == 0:
                continue

            self._receive_safe_protocol(worker_idx)

            self.worker_status[worker_idx] = 0

    def kill_straggler(self):
        cnt = 0
        for worker_idx in range(self.worker_num):
            if self.worker_status[worker_idx] == 0:
                # not running
                continue

            # check if straggler
            ok = self._receive_safe_protocol_tapcheck(worker_idx)
            if ok is not None:
                # not straggler
                self.worker_status[worker_idx] = 0
                continue

            # kill straggler
            self.workers[worker_idx].terminate()
            self.workers[worker_idx].join()
            self.worker_status[worker_idx] = 0
            self.pipes[worker_idx] = None
            cnt += 1

            # respawn
            parent_pipe, child_pipe = Pipe()
            self.pipes[worker_idx] = parent_pipe

            worker = Worker(name=self.name,
                            worker_idx=worker_idx,
                            pipe=child_pipe,
                            env_fn=self.env_fns[worker_idx],
                            policy=self.policy,
                            gamma=self.gamma,
                            max_sim_step=self.max_sim_step,
                            verbose=False,
                            device="cpu")
            worker.daemon = True

            self.workers[worker_idx] = worker
            worker.start()

        print(f"[POOL-MANAGER]: {cnt}/{self.worker_num} stragglers respawn")

    def close_pool(self):
        # print("[POOL-MANAGER] Start to shut down")
        self.wait_until_all_envs_idle()  # workers finish their jobs
        # print("[POOL-MANAGER] Wait for workers")
        for worker_idx in range(self.worker_num):
            self._send_safe_protocol(worker_idx, "KillProc", None)

        # print("[POOL-MANAGER] Joining all workers")
        for worker in self.workers:
            worker.join()
