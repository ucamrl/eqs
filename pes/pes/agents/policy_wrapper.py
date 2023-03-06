import random
import numpy as np

import torch
from torch.distributions.categorical import Categorical

# from Policy.PPO.PPOPolicy import PPOAtariCNN, PPOSmallAtariCNN


class PolicyWrapper:

    def __init__(self, policy_name: str, env_name: str, partial_cost_map: dict,
                 device):
        self.policy_name = policy_name
        self.partial_cost_map = partial_cost_map
        self.env_name = env_name
        self.device = device
        self.init_policy()

    def init_policy(self):
        if self.policy_name == "Random":
            self.policy_func = None
        elif self.policy_name == "Guide":
            assert self.partial_cost_map is not None, "Guide policy with None cost map"
            self.policy_func = None
        elif self.policy_name == "Greedy":
            assert self.partial_cost_map is not None, "Greedy policy with None cost map"
            self.policy_func = None

        # elif self.policy_name == "PPO":
        #     assert os.path.exists("./Policy/PPO/PolicyFiles/PPO_" +
        #                           self.env_name +
        #                           ".pt"), "Policy file not found"

        #     self.policy_func = PPOAtariCNN(
        #         self.action_n,
        #         device=self.device,
        #         checkpoint_dir="./Policy/PPO/PolicyFiles/PPO_" +
        #         self.env_name + ".pt")

        # elif self.policy_name == "DistillPPO":
        #     assert os.path.exists("./Policy/PPO/PolicyFiles/PPO_" +
        #                           self.env_name +
        #                           ".pt"), "Policy file not found"
        #     assert os.path.exists("./Policy/PPO/PolicyFiles/SmallPPO_" +
        #                           self.env_name +
        #                           ".pt"), "Policy file not found"

        #     full_policy = PPOAtariCNN(
        #         self.action_n,
        #         device="cpu",  # To save memory
        #         checkpoint_dir="./Policy/PPO/PolicyFiles/PPO_" +
        #         self.env_name + ".pt")

        #     small_policy = PPOSmallAtariCNN(
        #         self.action_n,
        #         device=self.device,
        #         checkpoint_dir="./Policy/PPO/PolicyFiles/SmallPPO_" +
        #         self.env_name + ".pt")

        #     self.policy_func = [full_policy, small_policy]

        else:
            raise NotImplementedError()

    def get_action(self, state, action_n: int):
        """for rollout"""
        if self.policy_name == "Random":
            return random.randint(0, action_n - 1)
        elif self.policy_name == "Guide":
            # use cost map as rollout logits
            logits = self.cost2logits(self.partial_cost_map[state])
            dist = Categorical(logits=logits)
            return int(dist.sample())
        elif self.policy_name == "Greedy":
            # use cost map greedily
            cost_list = self.partial_cost_map[state]
            idx = cost_list.index(min(cost_list))
            return idx

        # elif self.policy_name == "PPO":
        #     return self.categorical(self.policy_func.get_action(state))
        # elif self.policy_name == "DistillPPO":
        #     return self.categorical(self.policy_func[1].get_action(state))

        else:
            raise NotImplementedError()

    def get_value(self, state):
        if self.policy_name == "Random":
            return 0.0
        # elif self.policy_name == "PPO":
        #     return self.policy_func.get_value(state)
        # elif self.policy_name == "DistillPPO":
        #     return self.policy_func[0].get_value(state)
        else:
            raise NotImplementedError()

    def get_prior_prob(self, state, action_n: int):
        """for expansion selection"""
        if self.policy_name == "Random":
            return np.ones([action_n], dtype=np.float32) / action_n
        elif self.policy_name == "Guide":
            # use cost map as expansion logits
            if state is None:  # done
                return torch.ones([action_n], dtype=torch.float32) / action_n
            return self.cost2logits(self.partial_cost_map[state])
        elif self.policy_name == "Greedy":
            # use cost map as expansion logits
            if state is None:  # done
                return torch.ones([action_n], dtype=torch.float32) / action_n
            return self.cost2logits(self.partial_cost_map[state])

        # elif self.policy_name == "PPO":
        #     return self.policy_func.get_action(state)
        # elif self.policy_name == "DistillPPO":
        #     return self.policy_func[0].get_action(state)

        else:
            raise NotImplementedError()

    def cost2logits(self, logits: list[int]):
        s = sum(logits)
        ll = list(map(lambda x: s - x, logits))
        return torch.Tensor(ll)
