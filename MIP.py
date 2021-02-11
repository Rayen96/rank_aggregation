import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import numpy as np
import re
import scipy.stats as ss
import timeit
import matplotlib.pyplot as plt

pattern = '(\w)\[(\d+),(\d+)\]'



def MIP(data):
    
    n, m = data.shape

    better_than = np.zeros((n, n))
    equal_to = np.zeros((n, n))

    for i in range(data.shape[0]):
        for j in range(data.shape[0]):
            if i != j:
                better_than[i, j] = sum(data[i, k] < data[j, k] for k in range(
                    data.shape[1]) if data[i, k] is not None and data[j, k] is not None)
                equal_to[i, j] = sum(data[i, k] == data[j, k] for k in range(
                    data.shape[1]) if data[i, k] is not None and data[j, k] is not None)

    # Create a new model
    m = gp.Model("mip1")

    # Create variables
    x = m.addVars(n, n, vtype=GRB.BINARY, name="x")
    eq = m.addVars(n, n, vtype=GRB.BINARY, name="e")

    # Set objective
    m.setObjective(gp.quicksum(x[i, j]*(better_than[j, i]+equal_to[j, i]) + x[j, i]*(better_than[i, j]+equal_to[i, j])+(better_than[i, j]+better_than[j, i])
                            * eq[i, j] for i in range(n) for j in range(i)), GRB.MINIMIZE)

    # Add constraint: x + 2 y + 3 z <= 4
    m.addConstrs(x[i, j] + x[j, i] + eq[i, j] == 1 for i in range(n)
                for j in range(n) if i != j)

    # Add constraint: x + y >= 1
    m.addConstrs(x[i, k]-x[i, j]-x[j, k] >= -1 for i in range(n)
                for j in range(n) for k in range(n) if i != j != k)

    m.addConstrs(2*x[i, j]+2*x[j, i]+2*x[j, k]+2*x[k, j]-x[i, k]-x[k, i] >= 0 for i in range(n)
                for j in range(n) for j in range(n) for k in range(n) if i != j != k)

    # Optimize model
    m.optimize()
    res_list = {}
    res = np.empty((n, n))

    for v in m.getVars():
        res_list[v.varName] = v.x

    for i in res_list:
        letter, a, b = re.match(pattern, i).groups()
        if letter == 'x':
            res[int(a), int(b)] = res_list[i]


    aggregate = (res.sum(axis=0)+1)
    print(aggregate)
    print()
    print((aggregate == ss.rankdata(aggregate, method='min')).all())
    print('Obj: %g' % m.objVal)
    dist = kemeny(data, aggregate)
    print(f'dist: {dist}')
    print(f'Runtime: {m.runtime}')
    return m

def kendall_tau(pi, tau):
    dist = 0
    for i in range(len(pi)):
        for j in range(i):
            if (pi[i] is not None and pi[j] is not None and tau[i] is not None and tau[j] is not None):
                if  (pi[i] > pi[j] and tau[i] < tau[j]) or (pi[i] < pi[j] and tau[i] > tau[j]):
                    dist += 1
                elif (pi[i] == pi[j] and tau[i] != tau[j]) or (pi[i] != pi[j] and tau[i] == tau[j]):
                    dist += 1
    return dist


def kemeny(arr, agg):
    return sum(kendall_tau(arr[:, i], agg) for i in range(arr.shape[1]))


# rankings = np.random.randint(0, 30, size=(10, 6))
rankings = pd.read_pickle('compiled_main.pkl').sort_values('PR_rank')
rankings = rankings.iloc[:, :-1]
rankings = rankings.to_numpy()
print(rankings)

results = pd.DataFrame()
for i in range(5, 50):
    for j in range(1):
        
        m = MIP(rankings[:i, :])
        print(i, m.runtime)
        results = results.append({'n': i, 'experiment': j, 'runtime': m.runtime}, ignore_index=True)

results.to_csv('MIP_timing.csv', encoding='utf-8-sig')
results.boxplot(column='runtime', by='n')
plt.show()