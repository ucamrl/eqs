import functools
from rejoice.lib import Language, TestExprs
from rejoice import vars


class InfLang(Language):
    """A simple Lambda language for testing."""

    def get_supported_datatypes(self):
        return ["symbols"]

    @functools.cache
    def all_operators(self) -> "list[tuple]":
        return list(map(self.op_tuple, [
            ("f", "x")
        ]))

    @functools.cache
    def all_rules(self) -> "list[list]":
        x = vars("x") 
        y = "y"
        op = self.all_operators_obj()
        f = op.f

        return [
            ["grow",  f(x),  f(f(f(f(f(f(f(f(x))))))))],
            ["xy", f(x), f(y)],
            ["yx", f(y), f(x)],
            ["shrink", f(f(x)), f(x)]
        ]


    def get_terminals(self) -> "list":
        return []

    def eclass_analysis(self, car, cdr) -> any:
        return None

    def get_single_task_exprs(self):
        pass
        # ops = self.all_operators_obj()
        # f, g, h = ops.f, ops.g, ops.h

        # s = h(g("x", "Z"))
        # e = h(g("x", "Z"))

        # return TestExprs(saturatable=s,
        #                  explodes=e)

    def get_multi_task_exprs(self, count=16):
        """Get a list of exprs for use in multi-task RL training"""
        return [self.gen_expr(p_leaf=0.0) for i in range(count)]
