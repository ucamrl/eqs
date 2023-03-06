import os
import time
import random
import pickle
import datetime
import torch

from absl import app
from absl import flags

import gym
import numpy as np

from pes.agents.mcts_uct import MCTS_UCT
from pes.agents.wu_uct import WU_UCT
from pes.utils.utils import get_lang, load_expr

FLAGS = flags.FLAGS

# common
flags.DEFINE_string("lang", "MATH", "")
flags.DEFINE_string("env_id", "pes-base-env-v0", "must register")
flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_integer("depth_lim", 5, "expression depth limit")
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_integer("l", 1, "whether to log")
flags.DEFINE_integer("ver", 0, "whether to verbose")
flags.DEFINE_integer("early_stop", 10, "")

# E-graph
flags.DEFINE_integer("node_lim", 500, "enode limit")
flags.DEFINE_string("e", "greedy", "extractor; greedy or ilp")

# MCTS
flags.DEFINE_string("m", "mcts", "which model to use; either mcts or wu")
flags.DEFINE_integer("budget", 512, "search budget per planning step")
flags.DEFINE_integer("w1", 1, "expansion_worker_num")
flags.DEFINE_integer("w2", 16, "simulation_worker_num")
flags.DEFINE_integer("max_sim_step", 20,
                     "how long we rollouts if not terminating")
flags.DEFINE_string("p", "Random", "rollout policy; see policy wrapper")
flags.DEFINE_integer("max_depth", None, "max depth of MCTS, less important")
flags.DEFINE_integer(
    "max_width", None,
    "action space upper-bound of each node, if None, then no upperbound")
flags.DEFINE_float("gamma", 1., "if sparse reward, don't discounted?")


def reward_func(done: bool, info: dict, egraph, expr, base_cost):

    # per-step reward
    # if done:
    # else:
    # return reward

    # sparse reward;
    if done:
        cost, _ = egraph.extract(expr)
        reward = max(base_cost - cost, 0)
        info["actual_done"] = True
        info["ast_size"] = cost
    else:
        reward = 0
    return reward


def make_env(lang, expr, reward_func):

    def thunk():
        env = gym.make(FLAGS.env_id,
                       lang=lang,
                       expr=expr,
                       reward_func=reward_func,
                       node_lim=FLAGS.node_lim)
        return env

    return thunk


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)
    torch.manual_seed(FLAGS.seed)
    torch.backends.cudnn.deterministic = True

    # expression
    lang = get_lang(FLAGS.lang)()
    print("=" * 40)
    if FLAGS.fn is None:
        expr = lang.gen_expr(p_leaf=0., depth_limit=FLAGS.depth_lim)
        print("Generated expression: ", expr)
        print("Depth: ", FLAGS.depth_lim)
    else:
        fn = f"{FLAGS.default_out_path}/pes/inputs/"
        fn += FLAGS.fn
        expr = load_expr(lang, fn)
        print("Loaded expression: ", expr)
    print("=" * 40)

    # ===== env =====
    env_fn = make_env(lang, expr, reward_func)
    env = env_fn()
    state = env.reset()

    # ===== MCTS =====
    if FLAGS.m == "mcts":
        mcts = MCTS_UCT(env, FLAGS.budget, FLAGS.max_depth, FLAGS.max_width,
                        FLAGS.max_sim_step, FLAGS.gamma, None, FLAGS.p)
        assert (FLAGS.env_id == "pes-base-env-v0")
    elif FLAGS.m == "wu":
        mcts = WU_UCT(
            env, [make_env(lang, expr, reward_func) for i in range(FLAGS.w1)],
            [make_env(lang, expr, reward_func) for i in range(FLAGS.w2)],
            FLAGS.budget, FLAGS.max_depth, FLAGS.max_width, FLAGS.max_sim_step,
            FLAGS.gamma, FLAGS.w1, FLAGS.w2, FLAGS.p)
        assert (FLAGS.env_id == "pes-par-env-v0")
    else:
        raise RuntimeError(f"unsupported model: {FLAGS.m}")

    # ===== env loop =====
    verbose = bool(FLAGS.ver)
    cnt = 0
    sat_counter = 0
    episode_reward = 0
    game_start = time.perf_counter()
    infos = []
    while True:

        t0 = time.perf_counter()
        action = mcts.start_planning(state, verbose)
        t1 = time.perf_counter()
        action_space = len(env.get_action_space())
        next_state, reward, done, info = env.step(action)

        cnt += 1
        episode_reward += reward
        state = next_state

        stop_reason = info["stop_reason"]
        planning_t = t1 - t0
        info["planning_time"] = planning_t
        num_enodes = info["num_enodes"]
        num_eclasses = info["num_eclasses"]

        # NOTE: technically, mcts can only get sparse reward/extract
        # but for plot purpose, we extract every step
        if "ast_size" not in info:
            cost, _ = env.egraph.extract(env.expr)
            info["ast_size"] = cost

        infos.append(info)

        # NOTE: it may not really saturated, but can early stop
        if info["stop_reason"] == "SATURATED":
            sat_counter += 1
        else:
            sat_counter = 0

        print(f"iter {cnt} - reward {reward:.2f}", end=" - ")
        print(f"planning time {planning_t:.2f} seconds", end=" - ")
        print(f"stop reason {stop_reason}", end=" - ")
        print(f"AST size {cost}", end="; ")
        print(f"num_enodes {num_enodes}", end="; ")
        print(f"num_eclasses {num_eclasses}", end=" - ")
        print(f"Action {action}", end=" - ")
        print(f"action space: {action_space}")
        print("=" * 40)
        if done:
            break
        # early stop
        if sat_counter == FLAGS.early_stop:
            break

    mcts.close()
    game_end = time.perf_counter()
    print("DONE:: ")
    print(f"episode reward {episode_reward:.2f}", end=" - ")
    ast_size = info["ast_size"]
    print(f"extracted ast size {ast_size}", end=" - ")
    print(f"game time {game_end-game_start:.2f} seconds")
    print("=" * 40)

    # save
    log = bool(FLAGS.l)
    if log:
        print("[LOG]:: ")
        source = f"{FLAGS.lang}_gen" if FLAGS.fn is None else FLAGS.fn.split(
            ".")[0]
        # t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"{FLAGS.m}_{FLAGS.e}_{FLAGS.node_lim}_{source}"
        save_path = f"{FLAGS.default_out_path}/pes/runs/{run_name}"
        print("save path: ", save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # https://github.com/abseil/abseil-py/issues/57
        FLAGS.append_flags_into_file(save_path + "/hyperparams.txt")
        # MCTS-UCT has no learnable part,
        # so just save the action history
        with open(f"{save_path}/infos.pkl", "wb") as f:
            pickle.dump(infos, f)


if __name__ == "__main__":
    app.run(main)
