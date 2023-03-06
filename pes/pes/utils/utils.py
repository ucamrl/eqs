import time
import pickle
import pandas as pd
from collections import namedtuple

from pes.interface.MathLang import MathLang
from pes.interface.ToyLang import ToyLang
from pes.interface.PropLang import PropLang
from pes.interface.lib import Language, EGraph

step_info = namedtuple("StepInfo", [
    "action", "action_name", "stop_reason", "cost", "num_applications",
    "num_enodes", "num_eclasses", "best_expr", "init_expr", "extract_time"
])


def save_expr(exprs, path: str):
    with open(path, "wb") as f:
        pickle.dump(exprs, f)


def load_expr(lang, path: str) -> list:
    # this is needed to bring namedtuple to scope
    # NOTE if lang is not consistent, cannot load attribute
    lang.all_operators()
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data


def new_egraph(expr):
    egraph = EGraph()
    egraph.add(expr)
    return egraph


def add_df_meta(df: pd.DataFrame,
                lang_name: str,
                solver_name: str,
                base_cost,
                seed: int,
                node_lim: int,
                training_time=0.0):
    df["lang"] = lang_name
    df["base_cost"] = base_cost
    df["solver"] = solver_name
    df["seed"] = seed
    df["node_lim"] = node_lim
    if training_time is not None:
        df["training_time"] = training_time
    # add the step index as a column
    df = df.reset_index().rename(columns={"index": "step_ind"})
    return df


def get_lang(name: str) -> Language:
    return {
        "PROP": PropLang,
        "PropLang": PropLang,
        "MATH": MathLang,
        "MathLang": MathLang,
        "TOY": ToyLang,
    }[name]


def step(action: int, expr_to_extract, lang: Language, egraph: EGraph,
         node_lim):
    rw_rules = lang.rewrite_rules()
    rewrite_to_apply = [rw_rules[action]]
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(
        rewrite_to_apply, iter_limit=1, node_limit=node_lim)
    t0 = time.perf_counter()
    best_cost, best_expr = egraph.extract(expr_to_extract)
    t1 = time.perf_counter()
    # print("entry: ", egraph.extraction_entry_point(expr_to_extract))
    # print("all eid", egraph.eclass_ids())
    return step_info(action=action,
                     action_name=lang.rule_names[action],
                     num_applications=num_applications,
                     stop_reason=stop_reason,
                     cost=float(best_cost),
                     best_expr=str(best_expr),
                     num_eclasses=num_eclasses,
                     num_enodes=num_enodes,
                     init_expr=str(expr_to_extract),
                     extract_time=t1 - t0)
