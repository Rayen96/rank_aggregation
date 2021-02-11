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
    priority: float
    kemeny_value: float
    score_list: Any=field(compare=False)
    aggregate: Any=field(compare=False)


def generate_neighbor(pos, rankings_arr, aggregate, score_list):
    new_aggregate = aggregate.copy()
    new_scores = copy.deepcopy(score_list)
    rank = np.random.randint(len(aggregate))

    old_rank = aggregate[pos]

    kendall_helper.update_score_list(rankings_arr, new_aggregate, new_scores, pos, rank)
    new_aggregate[pos] = rank
    new_kemeny = kendall_helper.calculate_kemeny_distance(new_scores)
    
    return (Solution(-new_kemeny, new_kemeny, new_scores, new_aggregate), pos, rank)


def generate_neighbors(rankings_arr, aggregate, score_list, num_neighbors, short_term_memory):
    positions_to_change = np.random.choice(len(aggregate), num_neighbors)
    with Pool() as pool:
       heap =  pool.map(functools.partial(generate_neighbor, rankings_arr=rankings_arr, aggregate=aggregate, score_list=score_list), positions_to_change)
    heapq.heapify(heap)
    return heap


def choose_best_admissible_candidate(rankings_arr, candidates, current_solution, best_solution, short_term_memory, counter):
    
    for i in range(len(candidates)):
        candidate, pos, rank = heapq.heappop(candidates)
        if any(np.array_equal(candidate.aggregate, x) for x in short_term_memory):
            if candidate.kemeny_value > best_solution.kemeny_value:
                counter[pos, rank] += 1
                return candidate
        else:
            counter[pos, rank] += 1
            return candidate
    return None


def tabu(rankings_arr, aggregate, num_iter=1000, short_term_memory_size=500, elite_solutions_memory_size=10):
    
    short_term_memory = deque(maxlen=short_term_memory_size)
    elite_solutions = deque(maxlen=elite_solutions_memory_size)
    
    counter = np.zeros((len(aggregate), len(aggregate)))
    initial_score_list = kendall_helper.generate_score_list(rankings_arr, aggregate)
    initial_kemeny_distance = kendall_helper.calculate_kemeny_distance(initial_score_list)
    current_solution = Solution(-initial_kemeny_distance, initial_kemeny_distance, initial_score_list, aggregate)
    best_solution = copy.deepcopy(current_solution)
    short_term_memory.append(best_solution.aggregate)

    for i in range(num_iter):
        
        while True:
            candidates = generate_neighbors(rankings_arr, current_solution.aggregate, current_solution.score_list, 50, short_term_memory)

            current_solution = choose_best_admissible_candidate(rankings_arr, candidates, current_solution, best_solution, short_term_memory, counter)

            if current_solution is not None:
                break

        if current_solution.kemeny_value > best_solution.kemeny_value:
            best_solution = copy.deepcopy(current_solution)
            print(counter.sum(axis=1))

        print(best_solution.kemeny_value)

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
    tabu(rankings_arr, aggregate)