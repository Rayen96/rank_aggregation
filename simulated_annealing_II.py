import kendall_distance_helper
from dataclasses import dataclass
import numpy as np
import scipy.stats as ss
import mc4
import pandas as pd
import copy

@dataclass
class Aggregate:
    input_rankings: np.array
    rank: np.array
    counter: np.array
    distance: int


def initialize():
    data = pd.read_pickle('compiled_main.pkl').sort_values('PR_rank').iloc[:, :]
    res = data.iloc[:, :-1].to_numpy()
    agg = data.iloc[:, -1].to_numpy()
    counter = generate_counter(agg)
    print(data.shape)
    distance = kendall_distance_helper.kemeny_distance(res, agg)
    return Aggregate(res, agg, counter, distance)


def generate_counter(agg):
    counter = np.empty(len(agg)+1)
    unique =  np.unique(agg, return_counts=True)
    unique_lookup = {unique[0][i]: unique[1][i] for i in range(len(unique[0]))}
    for i in range(len(counter)):
        counter[i] = unique_lookup.get(i, 0)
    return counter


def split_tie_neighbor(aggregate, item):
    item_rank = aggregate.rank[item]

    aggregate.distance -= kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item)

    if aggregate.counter[item_rank] > 1:
        aggregate.counter[item_rank] -= 1
        aggregate.rank[item] += aggregate.counter[item_rank]
        aggregate.counter[aggregate.rank[item]] += 1

    aggregate.distance += kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item)


def merge_into_ties(aggregate, item):
    item_rank = aggregate.rank[item]

    aggregate.distance -= kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item)

    if aggregate.counter[item_rank] > 0 and item_rank > 1:
        aggregate.counter[item_rank] -= 1

        for new_rank in range(item_rank, -1, -1):
            if aggregate.counter[new_rank] > 0:
                break
        aggregate.rank[item] = new_rank
        aggregate.counter[new_rank] += 1

    aggregate.distance += kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item)


def swap(aggregate, item1, item2):

    aggregate.distance -= kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item1)
    temp = aggregate.rank[item1]
    aggregate.rank[item1] = aggregate.rank[item2]
    aggregate.distance += kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item1)

    aggregate.distance -= kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item2)
    aggregate.rank[item2] = temp
    aggregate.distance += kendall_distance_helper.kemeny_distance(aggregate.input_rankings, aggregate.rank, pair_update=item2)


def generate_neighbor(aggregate):
    n, m = aggregate.input_rankings.shape
    new = copy.deepcopy(aggregate)
    neighborhood_functions = [split_tie_neighbor, merge_into_ties, swap]

    chosen_function = np.random.choice(neighborhood_functions)
    if chosen_function.__name__ == 'swap':
        item1, item2 = np.random.choice(n, size=2, replace=False)
        chosen_function(new, item1, item2)
    else:
        item = np.random.randint(n)
        chosen_function(new, item)
    return new

def calculate_acceptance_probability(delta_E, T):
    return np.exp(-delta_E/T)

def calculate_adaptive_iterations(lb, fh, fl):
    F_ = 1 - np.exp(-(fh - fl)/fh)
    return int(lb + np.floor(lb*F_))

def new_temperature(T):
    return T*0.9

def simulated_annealing(Tmax, Tend):
    T = Tmax
    current_solution = initialize()
    n, m = current_solution.input_rankings.shape
    best_solution = copy.deepcopy(current_solution)
    L = n
    while T > Tend:
        acceptance_probability = []
        iterations_before_change = 0
        fh = 0
        fl = float('inf')
        print(L)
        for i in range(L):
            new_solution = generate_neighbor(current_solution)
            fh = max(fh, new_solution.distance)
            fl = min(fl, new_solution.distance)
            delta_E = new_solution.distance - current_solution.distance
            if delta_E <= 0:
                current_solution = new_solution
                if current_solution.distance < best_solution.distance:
                    best_solution = copy.deepcopy(current_solution)
            elif np.random.rand() < calculate_acceptance_probability(delta_E, T):
                acceptance_probability.append(calculate_acceptance_probability(delta_E, T))
                current_solution = new_solution
                if current_solution.distance < best_solution.distance:
                    best_solution = copy.deepcopy(current_solution)
        print(f'distance: {best_solution.distance}')
        L = calculate_adaptive_iterations(L, fh, fl)
        print(f'L: {L}')
        T = new_temperature(T)

if __name__ == '__main__':
    simulated_annealing(2.5, 0.01)