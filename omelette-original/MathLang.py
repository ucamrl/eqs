import functools
from rejoice.lib import Language, TestExprs
from rejoice import vars
import numpy as np


class MathLang(Language):
    """A simple Math language for testing."""

    def get_supported_datatypes(self):
        return ["integers"]

    @functools.cache
    def all_operators(self) -> "list[tuple]":
        return list(map(self.op_tuple, [
            ("Diff", "x", "y"),
            ("Integral", "x", "y"),
            ("Add", "x", "y"),
            ("Sub", "x", "y"),
            ("Mul", "x", "y"),
            ("Div", "x", "y"),
            ("Pow", "x", "y"),
            # ("Ln", "x"),
            ("Sqrt", "x"),

            ("Sin", "x"),
            ("Cos", "x")
        ]))

    @functools.cache
    def all_rules(self) -> "list[list]":
        a, b, c, x, f, g, y = vars("a b c x f g y") 
        op = self.all_operators_obj()
        return [
            ["comm-add", op.add(a, b), op.add(b, a)],
            ["comm-mul", op.mul(a, b), op.mul(b, a)],
            ["assoc-add", op.add(op.add(a, b), c), op.add(a, op.add(b, c))],
            ["assoc-mul", op.mul(op.mul(a, b), c), op.mul(a, op.mul(b, c))],

            ["sub-canon",  op.sub(a, b),  op.add(a, op.mul(-1, b))],
            ["zero-add",  op.add(a, 0),  a],
            ["zero-mul",  op.mul(a, 0),  0],
            ["one-mul",   op.mul(a, 1),  a],

            ["add-zero",  a,  op.add(a, 0)],
            ["mul-one",   a,  op.mul(a, 1)],

            ["cancel-sub",  op.sub(a, a),  0],

            ["distribute",  op.mul(a, op.add(b, c)), op.add(op.mul(a, b), op.mul(a, c))],
            ["factor",      op.add(op.mul(a, b), op.mul(a, c)),  op.mul(a, op.add(b, c))],
            ["pow-mul",  op.mul(op.pow(a, b), op.pow(a, c)),  op.pow(a, op.add(b, c))],
            ["pow1",     op.pow(x, 1),  x],
            ["pow2",     op.pow(x, 2),  op.mul(x, x)],

            ["d-add",  op.diff(x, op.add(a, b)),  op.add(op.diff(x, a), op.diff(x, b))],
            ["d-mul",  op.diff(x, op.mul(a, b)),  op.add(op.mul(a, op.diff(x, b)), op.mul(b, op.diff(x, a)))],

            ["d-sin",  op.diff(x, op.sin(x)),  op.cos(x)],
            ["d-cos",  op.diff(x, op.cos(x)),  op.mul(-1, op.sin(x))],

            ["i-one",    op.integral(1, x),          x],
            ["i-cos",    op.integral(op.cos(x), x),     op.sin(x)],
            ["i-sin",    op.integral(op.sin(x), x),     op.mul(-1, op.cos(x))],
            ["i-sum",    op.integral(op.add(f, g), x),  op.add(op.integral(f, x), op.integral(g, x))],
            ["i-dif",    op.integral(op.sub(f, g), x),  op.sub(op.integral(f, x), op.integral(g, x))],
            ["i-parts",  op.integral(op.mul(a, b), x),  op.sub(op.mul(a, op.integral(b, x)), op.integral(op.mul(op.diff(x, a), op.integral(b, x)), x))],            
        ]


    def get_terminals(self) -> "list":
        return [0, 1, 2]

    def eclass_analysis(self, car, cdr) -> any:
        ops = self.all_operators_obj()
        # This could be a literal encoded in a string
        try:
            return float(car)
        except:
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

    def get_single_task_exprs(self):
        ops = self.all_operators_obj()

        Add, Integral, Mul, Pow, Diff, Div, Cos, Sub, Sqrt = ops.add, ops.integral, ops.mul, ops.pow, ops.diff, ops.div, ops.cos, ops.sub, ops.sqrt

        s = ops.sub(ops.add(16, 2), 0)

        # s = Pow(x=Add(x=Ln(x=743), y=0), y=Sub(x=Sqrt(x=622), y=0))
        # s = Pow(x=Add(x=743, y=0), y=Sub(x=Sqrt(x=622), y=0))

        e = ops.mul(ops.add(16, 2), ops.mul(4, 0))

        return TestExprs(saturatable=s,
                         explodes=e)

    def get_multi_task_exprs(self, count=16):
        """Get a list of exprs for use in multi-task RL training"""
        return [self.gen_expr(p_leaf=0.0) for i in range(count)]

    def gen_expr(self, root_op=None, p_leaf=0.6, depth=0):
        """Generate an arbitrary expression which abides by the language."""
        depth_limit = 5
        ops = self.all_operators()
        root = np.random.choice(ops) if root_op is None else root_op
        children = []
        for i in range(len(root._fields)):
            if np.random.uniform(0, 1) < p_leaf or depth >= depth_limit:
                if np.random.uniform(0, 1) < 0.6:
                    children.append(np.random.choice(self.get_terminals()))
                else:
                    if "symbols" in self.get_supported_datatypes():
                        symbols = ["a", "b", "c", "d"]
                        # symbols = list(string.ascii_lowercase)
                        children.append(np.random.choice(symbols))
                    if "integers" in self.get_supported_datatypes():
                        children.append(np.random.randint(0, 3)) 
            else:
                chosen_op = np.random.choice(ops)
                op_children = []
                for j in range(len(chosen_op._fields)):
                    op_children.append(self.gen_expr(chosen_op, depth=depth+1))
                children.append(chosen_op(*op_children))
        return root(*children)
