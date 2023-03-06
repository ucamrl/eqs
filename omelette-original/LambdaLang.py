import functools
from rejoice.lib import Language, TestExprs
from rejoice import vars


class LambdaLang(Language):
    """A simple Lambda language for testing."""

    def get_supported_datatypes(self):
        return ["symbols"]

    @functools.cache
    def all_operators(self) -> "list[tuple]":
        return list(map(self.op_tuple, [
            ("f", "x"),
            ("g", "x", "y"),
            ("h", "x"),
            # ("add", "x", "y")
        ]))

    @functools.cache
    def all_rules(self) -> "list[list]":
        x, y = vars("x y") 
        Z = "Z"
        A = "A"
        op = self.all_operators_obj()
        f, g, h = op.f, op.g, op.h

        def rec(e, lim=100):
            if lim == 0:
                return e
            return rec(f(e), lim - 1)

        e = rec(x)

        # to reduce recursion depth needed
        # terms = ["a"*i for i in range(100)]
        # rules = [[t + "r", h(g(x, Z)), h(g(f(x), t))] for t in terms]

        return [
            ["inc",  h(g(x, Z)),  h(g(f(x), Z))],
            ["dec", g(f(x), y),  g(x, f(y))],
            ["clear", g(Z, y),     Z],
            # ["g3", e, x],  # resetting back to initial state

            # *rules

            # ["assoc-add", add(add(x, y), k), add(x, add(y, k))],
            # ["hx",  h(g(x, Z)),  h(g(f(x), W))],
            # ["g2x", g(W, y),     g(W, W)],

            # ["g4", g(Z, y),     g(y, Z)],
            # ["g5", g(y, Z),     g(Z, Z)],
            # ["g6", g(x, y), g(y, x)],
            # ["g7", g(y, x), g(x, y)],
            # ["h2", y,  f(f(f(f(y))))],
        ]


    def get_terminals(self) -> "list":
        return ["Z"]

    def eclass_analysis(self, car, cdr) -> any:
        return None

    def get_single_task_exprs(self):
        ops = self.all_operators_obj()
        f, g, h = ops.f, ops.g, ops.h

        s = h(g(f("x"), "Z")) # h(g("x", "Z"))
        e = h(g(f("x"), "Z"))

        return TestExprs(saturatable=s,
                         explodes=e)

    def get_multi_task_exprs(self, count=16):
        """Get a list of exprs for use in multi-task RL training"""
        return [self.gen_expr(p_leaf=0.0) for i in range(count)]
