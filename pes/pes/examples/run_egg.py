import os
import random
import datetime
import numpy as np
import pandas as pd
# from copy import deepcopy

from pes.interface.lib import Language
from pes.utils.utils import get_lang, new_egraph, add_df_meta, step, load_expr

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("node_lim", 500, "enode limit")
flags.DEFINE_integer("depth_lim", 5, "expression depth limit")
flags.DEFINE_integer("seed", 0, "")
flags.DEFINE_integer("l", 1, "whether to log")

flags.DEFINE_string("lang", "MATH", "")
flags.DEFINE_string("e", "greedy", "extractor; greedy or ilp")
flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("ver", 0, "verbose")


# TODO rewrite this to use the gym.env API
def solve_expr_egg(lang: Language, expr, node_lim):
    """
    Emulate egg's solver but WITHOUT an iteration limit.
    This will keep running until saturation,
        until a node limit, or time limit is reached.
    """
    egraph = new_egraph(expr)
    base_cost, _ = egraph.extract(expr)
    print("[EGG] base cost:", base_cost)

    steps = []

    i = 0
    sat_counter = 0
    verbose = bool(FLAGS.ver)

    while True:
        action_to_apply = i % lang.num_rules
        if action_to_apply == 0:
            sat_counter = 0

        result = step(action_to_apply, expr, lang, egraph, node_lim)
        steps.append(result)

        if verbose:
            print("=" * 40)
            print(result.stop_reason, result.num_applications,
                  result.num_enodes, result.num_eclasses)

        # normally it hits iter-limit and stop, thus apply rule one-step
        if result.stop_reason == 'NODE_LIMIT':
            print("***NODE limit***")
            break
        elif result.stop_reason == 'TIME_LIMIT':
            print("***TIME limit***")
            break  # egg stops optimizing
        elif result.stop_reason == 'SATURATED':
            sat_counter += 1

        if sat_counter == lang.num_rules:
            break  # egg has achieved saturation

        i += 1

    steps_df = pd.DataFrame(steps)
    steps_df = add_df_meta(steps_df, lang.name, "egg", base_cost, FLAGS.seed,
                           FLAGS.node_lim)

    print("=" * 40)
    print(f"[EGG] iter: {i}")
    # TODO add ILP extraction?
    print("greedy cost:", steps_df["cost"].iloc[-1])
    print("=" * 40)
    return steps_df


def main(_):
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    # load
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

    # egg solver
    print("=" * 40)
    print("[EGG] Solving expression", expr)
    print("=" * 40)
    egg_df = solve_expr_egg(lang, expr, FLAGS.node_lim)

    # save
    log = bool(FLAGS.l)
    if log:
        print("[LOG]:: ")
        source = f"{FLAGS.lang}_gen" if FLAGS.fn is None else FLAGS.fn.split(
            ".")[0]
        # t = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        run_name = f"egg_{FLAGS.e}_{FLAGS.node_lim}_{source}"
        save_path = f"{FLAGS.default_out_path}/pes/runs/{run_name}"
        print("save path: ", save_path)

        if not os.path.exists(save_path):
            os.mkdir(save_path)
        # https://github.com/abseil/abseil-py/issues/57
        FLAGS.append_flags_into_file(save_path + "/hyperparams.txt")
        egg_df.to_csv(f"{save_path}/egg.csv")


if __name__ == "__main__":
    app.run(main)
