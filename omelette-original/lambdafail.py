from ast import Lambda
from InfLang import InfLang

from rejoice import EGraph

from LambdaLang import LambdaLang

def egg_like(self, max_steps=500):
    egraph = self.new_egraph()
    steps = []
    for i in range(max_steps):
        s = Step(*self.step(egraph, (i % self.lang.num_rules)))
        steps.append(s)
        if s.stop_reason == 'NODE_LIMIT':
            print("hit node limit when searching...")
            if s.cost <= self.best_possible_cost:
                break
            else:
                raise Exception
    
    if s.cost > self.best_possible_cost:
        raise Exception("Failed to match egg's cost") 
    
    return steps

def lambda_saturate():
    node_limit = 10000
    lang = LambdaLang()
    ops = lang.all_operators_obj()
    f, g, h = ops.f, ops.g, ops.h
    Z = "Z"
    expr = h(g(Z, Z))
    egraph = EGraph()
    egraph.add(expr)
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(lang.rewrite_rules(), iter_limit=1000, node_limit=node_limit)
    best_cost, best_expr = egraph.extract(expr)
    print(expr, best_expr, best_cost, stop_reason, "enodes", num_enodes, "num_eclasses", num_eclasses)


    node_limit = 10000
    egraph = EGraph()
    egraph.add(expr)
    a1, a2, a3 = lang.rewrite_rules()
    actions = [a1]*3 + [a2, a3]*10 + [a1]

    for a in actions:
        stop_reason, num_applications, num_enodes, num_eclasses = egraph.run([a], iter_limit=1, node_limit=node_limit)

    best_cost, best_expr = egraph.extract(expr)
    print("optimal:")
    print(expr, best_expr, best_cost, stop_reason, "enodes", num_enodes, "num_eclasses", num_eclasses)

def inf_grow():
    node_limit = 10_000
    lang = InfLang()
    ops = lang.all_operators_obj()
    f = ops.f
    expr = f(f(f(f(f(f("x"))))))
    egraph = EGraph()
    egraph.add(expr)
    stop_reason, num_applications, num_enodes, num_eclasses = egraph.run(lang.rewrite_rules(), iter_limit=1000, node_limit=node_limit)
    best_cost, best_expr = egraph.extract(expr)
    print(expr, best_expr, best_cost, stop_reason, "enodes", num_enodes, "num_eclasses", num_eclasses)

    expr = f(f(f(f(f(f("x"))))))
    egraph = EGraph()
    egraph.add(expr)
    a1, a2, a3, a4 = lang.rewrite_rules()
    actions = [a1, a2] + [a2]*100
    for a in actions:
        stop_reason, num_applications, num_enodes, num_eclasses = egraph.run([a], iter_limit=1, node_limit=node_limit)

    best_cost, best_expr = egraph.extract(expr)
    print("optimal:")
    print(expr, best_expr, best_cost, stop_reason, "enodes", num_enodes, "num_eclasses", num_eclasses)

if __name__ == "__main__":
    lambda_saturate()

    # input: a a a a

    # a = a a a a a a a a
    # a a = a
    # ...
    # a = 1

    # input: f(f(f(f(x))))

    # f(x) = f(f(f(f(f(f(f(f(x))))))))
    # f(f(x)) = f(x)