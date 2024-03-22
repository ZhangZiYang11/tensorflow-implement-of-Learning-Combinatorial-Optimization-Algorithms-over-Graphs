import docplex.mp.model as cpx
import numpy as np

def mvc_opt(n, edges):#n为点的个数，edge为nparray，edge中点表示为从0到n-1

    set_I = range(n)#从0到n-1
    l = -0.5
    u = 1.5

    opt_model = cpx.Model(name="MIP Model")
    
    x_vars = {i: opt_model.integer_var(lb=l, ub=u, name="x_{0}".format(i)) for i in set_I}

    for edge in edges:
        opt_model.add_constraint(x_vars[edge[0]] + x_vars[edge[1]] <= 2)
        opt_model.add_constraint(x_vars[edge[0]] + x_vars[edge[1]] >= 1)
    
    objective = opt_model.sum(x_vars[i] for i in set_I)

    opt_model.minimize(objective)
    
    opt_model.solve()

    solution_list = []
    for v in opt_model.iter_integer_vars():
        # print(v, '=', v.solution_value)
        if v.solution_value == 1:
            solution_list.append(1)
        else:
            solution_list.append(0)
    
    return solution_list
