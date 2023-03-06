import time
from rejoice import EGraph
from rejoice.lib import Language
from rejoice.pretrain_dataset_gen import EGraphSolver
from PropLang import PropLang
import pandas as pd

def run_egg(lang: Language, expr, node_limit=10_000, iter_limit=7):
    print(f"running egg for expr", expr)
    first_stamp = int(round(time.time() * 1000))
    egraph = EGraph()
    egraph.add(expr)
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(lang.rewrite_rules(), iter_limit=iter_limit, node_limit=node_limit)
    print(stop_reason, "num_applications", num_applications, "num_enodes", num_enodes, "num_eclasses", num_eclasses)
    best_cost, best_expr = egraph.extract(expr)
    second_stamp = int(round(time.time() * 1000))
    # Calculate the time taken in milliseconds
    time_taken = second_stamp - first_stamp
    # egraph.graphviz("egg_best.png")
    print(f"egg best cost:", best_cost, "in",
            f"{time_taken}ms", "best expr: ", best_expr)
            
if __name__ == "__main__":
    lang = PropLang()
    ops = lang.all_operators_dict()
    AND, NOT, OR, IM = ops["and"], ops["not"], ops["or"], ops["implies"]
    x, y, z = "x", "y", "z"
    expr = OR(AND(x, y), IM(x, z))
    run_egg(lang, expr, 1_000, 7)
    solver = EGraphSolver(lang=lang, expr=expr, node_limit=1_000)
    steps = solver.egg_like(max_steps=10000)
    # steps = solver.optimize(max_steps=100000)
    print(steps)
    # solver.validate(actions=[0, 4, 7, 8, 6, 7, 9, 11, 2, 13, lang.num_rules])
    # steps = solver.analyze(max_steps=6500)
    # steps = pd.DataFrame(steps)
    # print(steps)