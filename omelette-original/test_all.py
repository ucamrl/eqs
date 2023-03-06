import argparse
import time
import os
from os import listdir
import random
import re
from collections import namedtuple
import torch
import numpy as np
import pandas as pd
from MathLang import MathLang
from PropLang import PropLang
from ppo import PPOAgent, run_ppo
from rejoice.lib import Language
from rejoice.rejoice import EGraph

Step = namedtuple("Step", ['action', 'action_name', 'stop_reason', 'cost', 'num_applications', 'num_enodes', 'num_eclasses', 'best_expr', 'init_expr'])

default_out_path = "dataset_metrics"

def new_egraph(expr):
    egraph = EGraph()
    egraph.add(expr)
    return egraph

def base_cost(expr):
    """Get the cost of the root expression in the initial egraph"""
    egraph = EGraph()
    egraph.add(expr)
    best_cost, _ = egraph.extract(expr)
    return best_cost

def step(action: int, expr_to_extract, lang: Language, egraph: EGraph, node_lim=10_000):
    rw_rules = lang.rewrite_rules()
    rewrite_to_apply = [rw_rules[action]]
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(rewrite_to_apply, iter_limit=1, node_limit=node_lim)
    best_cost, best_expr = egraph.extract(expr_to_extract)
    return Step(action=action,
                action_name=lang.rule_names[action],
                num_applications=num_applications,
                stop_reason=stop_reason,
                cost=float(best_cost),
                best_expr=str(best_expr),
                num_eclasses=num_eclasses,
                num_enodes=num_enodes,
                init_expr=str(expr_to_extract)
                )

def add_df_meta(df: pd.DataFrame, lang_name: str, solver_name: str, training_time=0.0):
    df["lang"] = lang_name
    df["solver"] = solver_name
    df["training_time"] = training_time
    # add the step index as a column
    df = df.reset_index().rename(columns={'index': 'step_ind'})
    return df

def solve_expr_egg(lang: Language, expr, node_lim=10_000):
    """
    Emulate egg's solver but WITHOUT an iteration limit.
    This will keep running until saturation, a node limit, or time limit is reached.
    """
    egraph = new_egraph(expr)
    best_cost, _ = egraph.extract(expr)
    print("base cost:", best_cost)

    steps = []

    i = 0
    sat_counter = 0

    while True:
        action_to_apply = i % lang.num_rules
        if action_to_apply == 0:
            sat_counter = 0

        result = step(action_to_apply, expr, lang, egraph, node_lim)
        steps.append(result)

        if result.stop_reason == 'NODE_LIMIT' or result.stop_reason == 'TIME_LIMIT':
            break  # egg stops optimizing
        elif result.stop_reason == 'SATURATED':
            sat_counter += 1

        if sat_counter == lang.num_rules:
            break  # egg has achieved saturation
        
        i += 1
    
    steps_df = pd.DataFrame(steps)
    steps_df = add_df_meta(steps_df, lang.name, "egg")
    return steps_df

def add_df_meta(df: pd.DataFrame, lang_name: str, solver_name: str, training_time=0.0):
    df["lang"] = lang_name
    df["solver"] = solver_name
    if training_time is not None:
        df["training_time"] = training_time
    # add the step index as a column
    df = df.reset_index().rename(columns={'index': 'step_ind'})
    return df

def solve_expr_rand(lang: Language, expr, time_lim, node_lim=10_000):
    """
    RANDOM agent.
    This will keep running until saturation, a node limit, or time limit is reached.
    """

    def run_once(lang, expr, max_ep_len=100):
        egraph = new_egraph(expr)
        steps = []
        count = 0
        while True:
            action = np.random.randint(0, lang.num_rules + 2)

            if action == lang.num_rules:
                break  # network has told us to take the end action
            elif action == lang.num_rules + 1:
                _, expr = egraph.extract(expr)
                egraph = new_egraph(expr)
            else:
                s = step(action, expr, lang, egraph, node_lim)
                steps.append(s)
                if s.stop_reason == 'NODE_LIMIT' or s.stop_reason == 'TIME_LIMIT':
                    break
                if count >= max_ep_len:
                    break
            count += 1

        if len(steps) == 0:
            return None  # ended immediately

        stepdf = pd.DataFrame(steps)
        return stepdf

    # run for as long as the agent was trained for.

    start = time.time()
    elapsed_time = 0
    time_lim = time_lim / 1000  # ms to s

    # Only interested in the best result found within the time limit.
    best_rollout_cost = np.inf
    best_rollout_len = np.inf
    best_rollout = None

    time_lim = 1

    print("running for", time_lim)
    while elapsed_time < time_lim:
        df = run_once(lang, expr)
        elapsed_time = time.time() - start
        if df is None or len(df) == 0:
            continue  # rollout ended immediately

        df["training_time"] = elapsed_time * 1000 # s to ms

        cost = df['cost'].iloc[-1]
        num_steps = len(df)

        same_cost_but_shorter = cost == best_rollout_cost and num_steps < best_rollout_len
        lower_cost = cost < best_rollout_cost

        if same_cost_but_shorter or lower_cost:
            best_rollout = df
            print(elapsed_time, 'new best', "c", cost, "s", num_steps, 'old', "c", best_rollout_cost, "s", best_rollout_len)
            best_rollout_cost = cost
            best_rollout_len = num_steps

    best_rollout = add_df_meta(best_rollout, lang.name, "random", training_time=None)

    print("finished running, took", elapsed_time)
    return best_rollout 
 

def rollout(lang: Language, expr, device, agent, training_time, num_rollouts=100, max_ep_len=100, node_lim=10_00, no_rebase=False):
    """Rollout an agent's trained policy on a given expression."""

    def run_once(lang, expr, device, agent, max_ep_len=100, node_lim=10_000, no_rebase=False):
        egraph = new_egraph(expr)
        steps = []
        agent.eval()
        count = 0
        while True:
            obs = lang.encode_egraph(egraph, use_shrink_action=True, step=count).to(device)
            with torch.no_grad():
                action, *rest = agent.get_action_and_value(obs, invalid_action_mask=obs.action_mask)

            action = action.item()
            if action == lang.num_rules:
                # print("end action received", action, "at", count)
                break  # network has told us to take the end action
            elif action == lang.num_rules + 1:
                # print("rebase action received", action, "at", count)
                if no_rebase:
                    continue  # no op
                else:
                    _, expr = egraph.extract(expr)
                    egraph = new_egraph(expr)
            else:
                s = step(action, expr, lang, egraph, node_lim)
                steps.append(s)
                if s.stop_reason == 'NODE_LIMIT' or s.stop_reason == 'TIME_LIMIT':
                    # print("node or time limit hit during policy rollout")
                    break  # should be rare
                if count >= max_ep_len:
                    break
            count += 1

        if len(steps) == 0:
            return None

        stepdf = pd.DataFrame(steps)
        return stepdf


   # PPO policy is stochastic, so try multiple times

    best_rollout_cost = np.inf
    best_rollout_len = np.inf
    best_rollout = None

    for i in range(num_rollouts):
        steps_df = run_once(lang, expr, device, agent, max_ep_len, node_lim, no_rebase)
        if steps_df is None or len(steps_df) == 0:
            continue  # rollout ended immediately

        cost = steps_df['cost'].iloc[-1]
        num_steps = len(steps_df)

        same_cost_but_shorter = cost == best_rollout_cost and num_steps < best_rollout_len
        lower_cost = cost < best_rollout_cost

        if same_cost_but_shorter or lower_cost:
            best_rollout = steps_df
            print('new best', "c", cost, "s", num_steps, 'old', "c", best_rollout_cost, "s", best_rollout_len)
            best_rollout_cost = cost
            best_rollout_len = num_steps

    solver_name = "norebase" if no_rebase else "omelette"

    best_rollout = add_df_meta(best_rollout, lang.name, solver_name, training_time=training_time)
    return best_rollout


def solve_expr_omelette(lang: Language, expr, expr_ind: int, egg_cost: int, egg_expr: str, node_lim=10_000, num_rollouts=100, max_ep_len=10, seed=1):
    """Train the PPO agent with its default config on this single expression in isolation."""
    print("Training agent...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    first_stamp = int(round(time.time() * 1000))
    agent = run_ppo(lang=lang.name,
                    seed=seed,
                    expr_str=str(expr),
                    exp_name=f"{lang.name}_{expr_ind}", 
                    node_limit=node_lim,
                    use_shrink_action=True,
                    learning_rate=5e-4,
                    max_episode_steps=max_ep_len,
                    total_timesteps=100_000,
                    egg_cost=egg_cost,
                    egg_expr=egg_expr,
                    max_cost=base_cost(expr),
                    print_actions=False)
    second_stamp = int(round(time.time() * 1000))
    training_time = second_stamp - first_stamp
    print("Agent trained. Evaluating learned policy...")

    df = rollout(lang=lang,
                 expr=expr,
                 device=device,
                 agent=agent,
                 training_time=training_time,
                 num_rollouts=num_rollouts,
                 max_ep_len=max_ep_len,
                 node_lim=node_lim)
    return df

def solve_expr(lang: Language, expr, expr_ind: int, node_lim=10_000, seed=1, out_path=default_out_path):
    print("Solving expression", expr)

    egg_df = solve_expr_egg(lang, expr, node_lim)
    print("egg cost:", egg_df["cost"].iloc[-1])
    egg_df.to_feather(f"{out_path}/{lang.name}_{expr_ind}_egg")

    om_df = solve_expr_omelette(lang=lang,
                                expr=expr,
                                expr_ind=expr_ind,
                                max_ep_len=100,
                                node_lim=node_lim,
                                egg_cost=egg_df["cost"].iloc[-1],
                                egg_expr=egg_df["best_expr"].iloc[-1],
                                seed=seed)
    om_df.to_feather(f"{out_path}/{lang.name}_{expr_ind}_om")


def get_lang(name: str) -> Language:
    return {
        "PROP": PropLang,
        "PropLang": PropLang,
        "MATH": MathLang,
        "MathLang": MathLang
    }[name]

def run_exps(lang_name: str, num_expr=10, node_lim=10_000, out_path=default_out_path, seed=1):
    # set random seeds for reproducability
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # create output dir if not exists
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    lang = get_lang(lang_name)()
    exprs = [(i, lang.gen_expr(p_leaf=0.0)) for i in range(num_expr)]
    # exprs = exprs[21:]
    # exprs = [exprs[2]]
    # exprs = [(0, lang.get_single_task_exprs().saturatable)]

    # filter expressions we already have in output dir
    # already_done_inds = [int(re.search(f'{lang.name}_(.+?)', file).group(1)) for file in listdir(out_path)]
    # print("already done", already_done_inds)
    # exprs = [i for j, i in enumerate(exprs) if j not in already_done_inds]

    for expr_ind, expr in exprs:
        # solve_expr(lang=lang, expr_ind=expr_ind, expr=expr, node_lim=node_lim, out_path=out_path, seed=seed)

        try:
            solve_expr(lang=lang, expr_ind=expr_ind, expr=expr, node_lim=node_lim, out_path=out_path, seed=seed)
        except:
            print("Failed to solve expr_ind", expr_ind)

    print("Completed running all experiments in generated dataset.")


def rerun_exps_om(lang_name: str, node_lim=10_000, max_ep_len=100, n_rollouts=100, out_path=default_out_path, seed=1, no_rebase=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True 

    # re-run omelette with the trained weights
    device = "cpu" # torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lang = get_lang(lang_name)()

    for file in listdir(out_path):
        fsplit = file.split("_")
        if fsplit[0] != lang.name or fsplit[2] != "om":
            continue

        expr_ind = fsplit[1] 
        df = pd.read_feather(f"{out_path}/{file}")

        agent_weights = torch.load(f"ppo_agent_weights/{lang.name}_{expr_ind}")
        agent = PPOAgent(n_actions=lang.num_rules + 2,
                         n_node_features=lang.num_node_features,
                         use_dropout=False,
                         use_edge_attr=True,
                         device=device)

        # load trained weights
        agent.load_state_dict(agent_weights)
        first = df.iloc[0]
        last = df.iloc[-1]
        expr = lang.eval_expr(first["init_expr"])
        print("no rebase", no_rebase)
        print("rerunning OM for expr", expr_ind, "om cost", last["cost"], "om steps", len(df))

        df = rollout(lang=lang,
                    expr=expr,
                    device=device,
                    agent=agent,
                    training_time=last["training_time"],
                    num_rollouts=100,
                    max_ep_len=max_ep_len,
                    node_lim=node_lim,
                    no_rebase=no_rebase)

        name = "norebase" if no_rebase else "omfixed"
        df.to_feather(f"{out_path}/{lang.name}_{expr_ind}_{name}")
        


def run_exps_rand(lang_name: str, node_lim=10_000, out_path=default_out_path, seed=1):
    """Runs a RANDOM agent on all tasks."""
    # for each expr .feather file in the generated folder,
    # load and pull omelette's training time
    # and keep expr_ind for use
    # and pull the first expr to pass to the runner
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    lang = get_lang(lang_name)()

    # filter expressions we already have in output dir
    already_done_inds = []
    # for file in listdir(out_path):
    #     fsplit = file.split("_")
    #     if fsplit[2] == "random" and fsplit[0] == lang.name:
    #         already_done_inds.append(fsplit[1])

    # print("already done", already_done_inds)

    for file in listdir(out_path):
        fsplit = file.split("_")
        if fsplit[0] != lang.name or fsplit[2] != "omfixed" or fsplit[1] in already_done_inds:
            continue

        expr_ind = fsplit[1] 
        df = pd.read_feather(f"{out_path}/{file}")
        first = df.iloc[0]
        last = df.iloc[-1]
        expr = lang.eval_expr(first["init_expr"])
        training_time = first["training_time"]
        print("running RANDOM for expr", expr_ind, "om cost", last["cost"], "om steps", len(df))
        random_df = solve_expr_rand(lang=lang, expr=expr, time_lim=training_time, node_lim=node_lim)
        random_df.to_feather(f"{out_path}/{lang.name}_{expr_ind}_random")



if __name__ == "__main__":
    node_lim = 500
    rerun_exps_om("MATH", node_lim=node_lim, seed=1, no_rebase=True)
    # run_exps_rand("PROP", node_lim=node_lim, seed=1)
    # run_exps("PROP", num_expr=100, node_lim=node_lim, seed=1)
    # run_exps("MATH", num_expr=25, node_lim=node_lim, seed=1)