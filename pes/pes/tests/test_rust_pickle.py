import os
import random
import pickle
import numpy as np

from pes.utils.utils import get_lang, new_egraph, step, load_expr

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_integer("node_lim", 500, "enode limit")
flags.DEFINE_integer("depth_lim", 5, "expression depth limit")
flags.DEFINE_integer("seed", 0, "")

flags.DEFINE_string("lang", "MATH", "")
flags.DEFINE_string("e", "greedy", "extractor; greedy or ilp")
flags.DEFINE_string("fn", None, "file name of the pre-generated expr")
flags.DEFINE_string("default_out_path", "data", "output dir")
flags.DEFINE_integer("ver", 0, "verbose")


def solve_expr_egg(lang, expr, node_lim):
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

    # NOTE:: Pickle and verify!
    num_enodes = egraph.num_enodes()
    num_eclasses = egraph.num_eclasses()
    total_size = egraph.total_size()
    eclass_ids = egraph.eclass_ids()
    entry_point = egraph.extraction_entry_point(expr)[0]

    path = os.path.join(FLAGS.default_out_path, "egraph.pkl")
    with open(path, "wb") as f:
        pickle.dump(egraph, f)
    print("Egraph pickle OK")

    with open(path, "rb") as f:
        load_egraph = pickle.load(f)
    print("Egraph un-pickle OK")

    assert (load_egraph.num_enodes() == num_enodes)
    assert (load_egraph.num_eclasses() == num_eclasses)
    assert (load_egraph.total_size() == total_size)
    assert (load_egraph.extraction_entry_point(expr)[0] == entry_point)

    load_id = load_egraph.eclass_ids()
    for i, eid in enumerate(eclass_ids):
        assert (eid == load_id[i])


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
    solve_expr_egg(lang, expr, FLAGS.node_lim)


if __name__ == "__main__":
    # NOTE test pickleme
    from rust_lib import PickleMe
    print()
    print("pickle me")
    print()

    print("simple test")
    tmp = PickleMe()
    print(tmp.foo)
    tmp.foo = 3
    print(tmp.foo)
    print(tmp.get_attr())

    tmp = PickleMe(1)
    print(tmp.get_attr())
    print("pickle test")
    tmp = pickle.loads(pickle.dumps(tmp))
    print(tmp.get_attr())
    print("bincode test")
    print("pickle test incomplete")
    print()
    print("pickle me done")
    print()

    app.run(main)
