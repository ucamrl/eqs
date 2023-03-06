import random

from absl import app
from absl import flags

import numpy as np

from pes.utils.utils import get_lang, save_expr

FLAGS = flags.FLAGS
flags.DEFINE_string("lang", "MATH", "")
flags.DEFINE_integer("depth_lim", 5, "depth limit of expression tree")

flags.DEFINE_integer("seed", 42, "")
flags.DEFINE_string("default_out_path", "data", "output dir")


def main(_):
    # set random seeds for reproducability
    random.seed(FLAGS.seed)
    np.random.seed(FLAGS.seed)

    lang = get_lang(FLAGS.lang)()
    expr = lang.gen_expr(p_leaf=0., depth_limit=FLAGS.depth_lim)

    # file name
    tmp_file = f"{FLAGS.default_out_path}/pes/inputs/"
    tmp_file += f"{FLAGS.lang}-{FLAGS.depth_lim}-{FLAGS.seed}.pkl"

    # ==========================
    print("expr::")
    print(expr)

    # save is OK
    save_expr(expr, tmp_file)
    print("Save OK")


if __name__ == "__main__":
    app.run(main)
