import docplex.mp.model as cpx
import numpy as np

def maxcut_opt(n, edges, weights):#n为点的个数，edges为边，weights为边的权重，一一对应
    
    set_I = range(n)#从0到n-1
    l = -0.5
    u = 1.5

    opt_model = cpx.Model(name='MIP Model')

    x_vars = {i: opt_model.integer_var(lb=l, ub=u, name="x_{0}".format(i)) for i in set_I}

    objective = opt_model.sum((x_vars[edges[k][0]]-x_vars[edges[k][1]])*(x_vars[edges[k][0]]-x_vars[edges[k][1]]) * weights[k] for k in range(len(edges)))

    opt_model.maximize(objective)

    opt_model.solve()

    solution_list=[]
    for v in opt_model.iter_integer_vars():
        # print(v, '=', v.solution_value)
        if v.solution_value == 1:
            solution_list.append(1)
        else:
            solution_list.append(0)
    
    return solution_list

# list1 = maxcut_opt(10, np.array([[1,3],[1,4],[4,5]]), np.array([10,100,20]))
# print(list1)