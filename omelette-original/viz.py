import unittest

import torch

from rejoice import *
from rejoice import envs, EGraph
from rejoice.lib import Language
from PropLang import PropLang
import gym


class VizTests(unittest.TestCase):

    def setUp(self) -> None:
        self.lang = PropLang()
        ops = self.lang.all_operators_dict()
        AND, NOT, OR, IM = ops["and"], ops["not"], ops["or"], ops["implies"]
        x, y, z = "x", "y", "z"
        # self.expr = AND(x, True)  # AND(IM(NOT(y), NOT(x)), IM(y, z))
        self.expr = AND(IM(NOT(y), NOT(x)), IM(y, z))
        self.egraph = EGraph()

    # def test_node_feature_count(self):
    #     self.assertEqual(self.lang.num_node_features(), 7)

    def test_egraph_encode(self):
        self.egraph.add(self.expr)
        self.egraph.graphviz("./test.png")
        classes = self.egraph.classes()
        data = self.lang.encode_egraph(self.egraph)
        # data2 = self.lang.encode_egraph(self.egraph)
        # There should be num_enodes + num_eclasses nodes total
        # visual confirmation
        self.lang.viz_egraph(data)
        # self.lang.viz_egraph(data2)

    # def test_rw_rules(self):
    #     """Ensure egg creates a rewrite rule for each one defined in our lang"""
    #     self.assertEqual(len(self.lang.rewrite_rules()),
    #                      len(self.lang.all_rules()))

    # # The test below is useful for debugging, but it outputs images of the graphs for
    # # verification. Therefore only uncomment and run when you need to.
    # def test_egg_rw(self):
    #     egraph1 = EGraph()
    #     egraph1.add(self.expr)
    #     egraph1.graphviz("./egraph1_before.png")
    #     data1_before = self.lang.encode_egraph(egraph1)
    #     egraph1.run(self.lang.rewrite_rules(), 5)
    #     data1_after = self.lang.encode_egraph(egraph1)
    #     egraph1.graphviz("./egraph1_after.png")
    #     self.lang.viz_egraph(data1_before)
    #     self.lang.viz_egraph(data1_after)
    #     print(data1_after.x)

    #     egraph2 = EGraph()
    #     egraph2.add(self.expr)
    #     egraph2.graphviz("./egraph2_before.png")
    #     data2_before = self.lang.encode_egraph(egraph2)
    #     egraph2.run(self.lang.rewrite_rules(), 5)
    #     data2_after = self.lang.encode_egraph(egraph2)
    #     egraph2.graphviz("./egraph2_after.png")
    #     self.lang.viz_egraph(data2_before)
    #     self.lang.viz_egraph(data2_after)

    # def run_egg(self):
    #     eg = EGraph()
    #     eg.add(self.expr)
    #     eg.run(self.lang.rewrite_rules(), 7)
    #     best_cost, best_expr = eg.extract(self.expr)
    #     print("egg", self.lang.encode_egraph(eg))
    #     print("egg best cost:", best_cost, "best expr: ", best_expr)

    # def take_step(self, env, action):
    #     new_obs, r, done, _ = env.step(action)  # mul-0
    #     print("Obs:", new_obs, "Reward:", r, "Done:", done)

    # def test_env(self):
    #     self.run_egg()
    #     env = gym.make('egraph-v0', lang=self.lang, expr=self.expr)
    #     init_obs = env.reset()
    #     print("Init_Obs", init_obs)
    #     self.take_step(env, 3)
    #     self.take_step(env, 3)
    #     self.take_step(env, 4)
    #     self.take_step(env, 5)
    #     reset_obs = env.reset()
    #     print("Reset_Obs", reset_obs)
    #     self.take_step(env, 3)
    #     self.take_step(env, 3)
    #     self.take_step(env, 4)
    #     self.take_step(env, 5)

    # def test_r(self):
    #     self.run_egg()
    #     env = gym.make('egraph-v0', lang=self.lang, expr=self.expr)
    #     init_obs = env.reset()
    #     print("Init_Obs", init_obs)
    #     self.take_step(env, 3)
    #     self.take_step(env, 0)
    #     self.take_step(env, 2)
    #     self.take_step(env, 2)
    #     self.take_step(env, 3)
    #     self.take_step(env, 0)
    #     self.take_step(env, 1)


if __name__ == '__main__':
    unittest.main()
