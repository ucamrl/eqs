import functools
from pes.interface.lib import Language

from rust_lib import vars


class ToyLang(Language):
    """re-implement the colab example"""

    def get_supported_datatypes(self):
        return ["integers"]

    @functools.cache
    def all_operators(self) -> list[tuple]:
        return list(
            map(self.op_tuple, [
                ("add", "x", "y"),
                ("times", "x", "y"),
                ("shift", "x", "y"),
                ("div", "x", "y"),
            ]))

    @functools.cache
    def all_rules(self) -> list[list]:
        a, b, c, x, f, g, y, z = vars("a b c x f g y z")
        op = self.all_operators_obj()
        return [
            ["time-is-shift", op.times(x, 2),
             op.shift(x, 1)],
            [
                "re-associate",
                op.div(op.times(x, y), z),
                op.times(x, op.div(y, z))
            ],
            ["simplify-div", op.div(x, x), 1],
            ["simplify-times", op.times(x, 1), x],
        ]

    def get_terminals(self) -> list[int]:
        return [0, 1, 2, 3]

    def eclass_analysis(self, car, cdr) -> any:
        ops = self.all_operators_obj()
        # This could be a literal encoded in a string
        try:
            return float(car)
        except:
            print("analysis fail")
            pass

        # Else it is an operation with arguments
        op = car
        args = cdr

        try:
            a = float(args[0])
            b = float(args[1])
            if op == ops.add:
                return a + b
            if op == ops.sub:
                return a - b
            if op == ops.mul:
                return a * b
            if op == ops.div and b != 0.0:
                return a / b
        except:
            pass
        return None

    def gen_expr(self, p_leaf=0.0, depth_limit=0):
        """the colab example
        symbol a is not yet support"""
        opss = self.all_operators_obj()
        expr = opss.div(opss.times(3, 2), 2)
        return expr

    def get_op_tbl(self) -> dict:
        return self.op_to_ind

    def get_term_tbl(self) -> dict:
        # FIXME
        terminals = self.get_terminals()
        return {item: i for i, item in enumerate(terminals)}
