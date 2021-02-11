import numpy as np
import pandas as pd
from collections import namedtuple
import scipy.stats as ss
import kendall_helper
from mc4 import MC4
from data_reader import read_df
from collections import deque
from dataclasses import dataclass, field
from typing import Any
import heapq
import copy
from multiprocessing import Pool
import functools
from collections import Counter

@dataclass(order=True)
class Solution:
    kemeny_value: float
    score_list: Any=field(compare=False)
    aggregate: Any=field(compare=False)


def generate_neighbor(rankings_arr, solution):
    new_aggregate = solution.aggregate.copy()
    new_scores = copy.deepcopy(solution.score_list)

    rank = np.random.randint(len(aggregate))
    pos = np.random.randint(len(aggregate))

    kendall_helper.update_score_list(rankings_arr, new_aggregate, new_scores, pos, rank)
    new_aggregate[pos] = rank
    new_kemeny = kendall_helper.calculate_kemeny_distance(new_scores)
    
    return Solution(new_kemeny, new_scores, new_aggregate)


def simulated_annealing(rankings_arr, aggregate, Tmax, Tend):
    T = Tmax
    initial_score_list = kendall_helper.generate_score_list(rankings_arr, aggregate)

    initial_kemeny_distance = kendall_helper.calculate_kemeny_distance(initial_score_list)
    current_solution = Solution(initial_kemeny_distance, initial_score_list, aggregate)
    best_solution = copy.deepcopy(current_solution)

    while T > Tend:
        for i in range(500):
            new_solution = generate_neighbor(rankings_arr, current_solution)
            delta_E = -(new_solution.kemeny_value - current_solution.kemeny_value)

            if delta_E <= 0:
                current_solution = new_solution

            elif np.random.rand() < np.exp(-delta_E/T):
                current_solution = new_solution
            if current_solution.kemeny_value > best_solution.kemeny_value:
                best_solution = current_solution
                print(best_solution.kemeny_value)
        T *= 0.99

if __name__ == '__main__':
    rankings_df = read_df('../merged_2021-01-19.csv')
    aggregate_df = MC4(rankings_df)
    aggregate = aggregate_df['PR_rank']

    rankings_df['PR_initial'] = aggregate

    rankings_df = rankings_df.sort_values('PR_initial')

    rankings_df = rankings_df.iloc[:500, :]

    print(rankings_df)
    aggregate = np.array(rankings_df['PR_initial'])

    rankings_arr = rankings_df.iloc[:, :-1].to_numpy()
    simulated_annealing(rankings_arr, aggregate, 0.02, 0.00001)