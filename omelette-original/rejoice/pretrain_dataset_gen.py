from collections import namedtuple
from rejoice.util import time_limit
from .PretrainingDataset import PretrainingDataset
import os
import time
import torch
from .rejoice import EGraph
from .lib import Language
import numpy as np
import sys
from pathlib import Path
import itertools
import uuid


Step = namedtuple("Step", ['action', 'action_name', 'stop_reason', 'cost', 'num_applications', 'num_enodes', 'num_eclasses', 'best_expr'])

class EGraphSolver:

    def __init__(self, lang: Language, expr: any, node_limit=10_000, rng=np.random.default_rng(), iter_lim=7):
        self.lang = lang
        self.expr = expr
        self.rewrite_rules = lang.rewrite_rules()
        self.node_limit = node_limit
        self.iter_limit = iter_lim
        self.rng = rng
        self.max_cost, _ = self.new_egraph().extract(self.expr)
        self.best_possible_cost, self.best_possible_expr, self.num_applications, self.num_enodes, self.num_eclasses = self.exhaustive_search()

    def egg_like(self, max_steps=500):
        egraph = self.new_egraph()
        steps = []
        for i in range(max_steps):
            s = Step(*self.step(egraph, (i % self.lang.num_rules)))
            steps.append(s)
            if s.stop_reason == 'NODE_LIMIT':
                print("hit node limit when searching...")
                if s.cost <= self.best_possible_cost:
                    break
                else:
                    raise Exception
        
        if s.cost > self.best_possible_cost:
            raise Exception("Failed to match egg's cost") 
        
        return steps


    def optimize(self, max_steps=500):
        # print("egg cost:", best_possible_cost, "expr", best_possible_expr)
        # try to re-create egg's cost, this time tracking the actions at each step
        egraph = self.new_egraph()
        steps = []
        for i in range(max_steps):
            s = Step(*self.step(egraph, (i % self.lang.num_rules)))
            if s.stop_reason == 'NODE_LIMIT':
                print("hit node limit when searching...")
                steps.append(s)
                if s.cost <= self.best_possible_cost:
                    break
                else:
                    raise Exception
            elif s.stop_reason != 'SATURATED':  # if 'SATURATED', the step didn't change the egraph; we can filter it
                steps.append(s)
                if s.cost <= self.best_possible_cost:
                    break  # we don't need to take any more steps; we found the min cost
        
        if s.cost > self.best_possible_cost:
            raise Exception("Failed to match egg's cost")
            
        
        print("matched egg cost. Finding minimum sequence...")
        # print(steps)
        # now extract the minimum action order needed to get this cost
        all_poss_seqs = list(itertools.product([0, 1], repeat=len(steps)))
        # sorting by sum means that we try the smallest action sequence lengths first
        all_poss_seqs.sort(key=sum)
        egraph_base = self.new_egraph()
        actions_needed: list[int] = []
        done = False
        action_list: list[int] = [step.action for step in steps]
        print(action_list)

        min_steps: list[Step] = []
    
        for seq_mask in all_poss_seqs:
            eg = egraph_base.clone()
            actions = list(itertools.compress(action_list, seq_mask))
            min_steps = []
            for ind, action in enumerate(actions):
                s = Step(*self.step(eg, action))
                min_steps.append(s)
                # stop_reason = eg.run(
                #     [self.rewrite_rules[action]], iter_limit=1, node_limit=self.node_limit)
                # if stop_reason != 'NODE_LIMIT':
                # cost, ex = eg.extract(self.expr)
                if s.cost <= self.best_possible_cost:  # found the shortest action sequence to achieve cost equiv to egg
                    actions_needed = actions[:ind+1]
                    actions_needed.append(self.lang.num_rules)
                    print("found best", s.cost, s.best_expr, actions_needed)
                    done = True
                    break
                # else:
                #     print("Exceeded NODE_LIMIT during sequence gen")
                #     break
            if done:
                break

        self.build_pyg_data(actions_needed)
        return min_steps

    def analyze(self, max_steps=500):
        """Find the minimum action sequence which causes the e-node limit to be exceeded."""
        # try to re-create egg's cost, this time tracking the actions at each step
        egraph = self.new_egraph()
        Step = namedtuple("Step", ['action', 'action_name', 'stop_reason', 'cost', 'num_applications', 'num_enodes', 'num_eclasses'])
        steps = []
        for i in range(max_steps):
            action, stop_reason, cost, num_applications, num_enodes, num_eclasses = self.step(egraph, (i % self.lang.num_rules))
            action_name = self.lang.all_rules()[action][0]
            step = Step(action, action_name, stop_reason, cost, num_applications, num_enodes, num_eclasses)
            if stop_reason == 'NODE_LIMIT':
                print("hit node limit when searching...")
                steps.append(step)
                break
            elif stop_reason != 'SATURATED':  # if 'SATURATED', the step didn't change the egraph; we can filter it
                steps.append(step)

    #     # create minimal sequence
    #     actions = [step.action for step in steps]
    #     all_poss_seqs = list(itertools.product([0, 1], repeat=len(actions)))
    #    # sorting by sum means that we try the smallest action sequence lengths first
    #     all_poss_seqs.sort(key=sum)
    #     egraph_base = self.new_egraph()
    #     actions_needed = []
    #     done = False

    #     for seq_mask in all_poss_seqs:
    #         eg = egraph_base.clone()
    #         actions_masked = list(itertools.compress(actions, seq_mask))  # mask
    #         print(actions_masked)
    #         for ind, action in enumerate(actions_masked):
    #             stop_reason = eg.run(
    #                 [self.rewrite_rules[action]], iter_limit=1, node_limit=self.node_limit)
    #             if stop_reason == 'NODE_LIMIT':
    #                 print("Exceeded NODE_LIMIT during sequence gen")
    #                 actions_needed = actions_masked[:ind+1]
    #                 done = True
    #                 break
    #         if done:
    #             break        
    #     print(actions_needed)

        
        return steps

    def build_pyg_data(self, actions: "list[int]"):
        """Convert an action sequence to a list of PyTorch Geometric data objects."""
        lang_name = self.lang.name
        if not os.path.exists(lang_name):
            os.makedirs(lang_name)

        egraph = self.new_egraph()
        egid = uuid.uuid4().hex

        for ind, action in enumerate(actions):
            data = self.lang.encode_egraph(egraph, action)
            data.max_cost = self.max_cost
            data.min_cost = self.best_possible_cost
            # number of actions taken by egg to solve the expression this graph is from
            data.egg_rewrites = self.num_applications

            data.egg_enodes = self.num_enodes
            data.egg_eclasses = self.num_eclasses
            torch.save(
                data, f'{lang_name}/{egid}_{ind}.pt')
            if action < self.lang.num_rules:
                egraph.run([self.lang.rewrite_rules()[action]],
                           iter_limit=1, node_limit=self.node_limit)

    def validate(self, actions):
        egraph = self.new_egraph()
        for action in actions:
            if action < self.lang.num_rules:
                egraph.run([self.lang.rewrite_rules()[action]],
                           iter_limit=1, node_limit=self.node_limit)
            elif action == self.lang.num_rules:
                self.exhaustive_search(self.iter_limit)
                best_cost, best_expr = egraph.extract(self.expr)
                if best_cost == self.best_possible_cost:
                    print("Validated")
                else:
                    print("Failed to validate")
                    raise Exception("Validation failed for egraph")

    def step(self, egraph: EGraph, action: int):
        rewrite_to_apply = [self.rewrite_rules[action]]
        stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(
            rewrite_to_apply, iter_limit=1, node_limit=self.node_limit)
        best_cost, best_expr = egraph.extract(self.expr)
        best_cost = float(best_cost)
        return action, self.lang.all_rules()[action][0], stop_reason, best_cost, num_applications, num_enodes, num_eclasses, best_expr

    def exhaustive_search(self, iter_limit=None):
        if iter_limit is None:
            iter_limit = self.iter_limit
        egraph = self.new_egraph()
        stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(self.lang.rewrite_rules(), iter_limit=iter_limit,
                   node_limit=self.node_limit, use_backoff=True)
        best_cost, best_expr = egraph.extract(self.expr)
        best_cost = float(best_cost)
        return best_cost, best_expr, num_applications, num_enodes, num_eclasses

    def new_egraph(self):
        egraph = EGraph()
        egraph.add(self.expr)
        return egraph


def generate_dataset(lang: Language, num=10, rng=np.random.default_rng()):
    exprs = [lang.gen_expr(p_leaf=0.0) for i in range(num)]

    for ind, expr in enumerate(exprs):
        print("Generating expr", ind, expr)
        solver = EGraphSolver(lang, expr)
        try:
            solver.optimize()
        except Exception as e:
            print("Failed to solve expr", ind, expr)
            print(e)
            continue

def generate_exploders(lang: Language, num=1000, rng=np.random.default_rng()):
    exprs = [lang.gen_expr(p_leaf=0.0) for i in range(num)]
    for ind, expr in enumerate(exprs):
        print("Generating expr", ind, expr)
        # attempt to solve each expression with a 20k iteration limit. 