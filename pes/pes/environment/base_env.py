import gym
from gym import spaces
from collections import namedtuple

from pes.interface.lib import EGraph

CKPT = namedtuple("checkpoint",
                  ["egraph", "sat_counter", "base_cost", "count"])


class BaseEnvironment(gym.Env):

    def __init__(self, lang, expr, reward_func, node_lim):
        super().__init__()
        self.lang = lang
        self.expr = expr
        self.reward_func = reward_func
        self.rules = lang.rewrite_rules()
        self.num_rules = len(self.rules)
        self.node_lim = node_lim
        self.egraph = None
        self.base_cost = 0
        self.cnt = 0
        self.sat_counter = 0

        # ===== gym.Env attr =====
        self.observation_space = spaces.Discrete(1)  # no use
        self.action_space = spaces.Discrete(self.num_rules)  # no use
        # self.reward_range = ?
        # ===== gym.Env attr =====

    def reset(self):
        # reset egraph
        self.egraph = EGraph()
        self.egraph.add(self.expr)
        self.base_cost, _ = self.egraph.extract(self.expr)
        self.cnt = 0
        self.sat_counter = 0
        return self._build_state()

    def get_action_space(self) -> list[int]:
        return [i for i in range(self.num_rules)]

    def step(self, action: int):
        assert (self.egraph is not None), "None Egraph"
        self.cnt += 1
        rewrite_to_apply = [self.rules[action]]
        stop_reason, num_applications, num_enodes, num_eclasses = self.egraph.run(
            rewrite_to_apply, iter_limit=1, node_limit=self.node_lim)

        done = False
        info = {
            "stop_reason": stop_reason,
            "count": self.cnt,
            "num_applications": num_applications,
            "num_enodes": num_enodes,
            "num_eclasses": num_eclasses,
            "action": action,
            "action_name": self.lang.rule_names[action]
        }
        if stop_reason == "NODE_LIMIT" or stop_reason == "TIME_LIMIT":
            done = True
            self.sat_counter = 0

        # SATURATED means using this rule add nothing to the egraph
        # XXX this is not sufficient!
        elif stop_reason == "SATURATED":
            self.sat_counter += 1
            if self.sat_counter == self.num_rules:
                done = True
        else:
            self.sat_counter = 0

        reward = self.reward_func(done, info, self.egraph, self.expr,
                                  self.base_cost)
        next_state = self._build_state()
        return next_state, reward, done, info

    def _build_state(self):
        # assert (False), "to be override"
        return None

    def checkpoint(self):
        # NOTE: MUST deepcopy
        return CKPT(egraph=self.egraph.clone(),
                    sat_counter=self.sat_counter,
                    base_cost=self.base_cost,
                    count=self.cnt)

    def restore(self, ckpt):
        assert (isinstance(ckpt, CKPT)), "ckpt?"
        # NOTE: MUST deepcopy
        self.egraph = ckpt.egraph.clone()
        self.sat_counter = ckpt.sat_counter
        self.base_cost = ckpt.base_cost
        self.cnt = ckpt.count
