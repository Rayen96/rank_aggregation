import pandas as pd
import numpy as np
import scipy.stats as ss
from kendall_helper import confirm_kemeny

def rescale(rankings_arr):
    size = (~np.isnan(rankings_arr)).sum(axis=0)
    new = np.empty(rankings_arr.shape)
    order = np.argsort(-size)
    new[:, order[0]] = rankings_arr[:, order[0]]
    for i in range(1, len(order)):
        current = np.take(rankings_arr, order[i], axis=1)
        prev = np.take(rankings_arr, order[:i], axis=1)
        ascending_row = np.argsort(current)
        descending_row = np.argsort(-current)
        
        for min_idx in ascending_row:
            avg_min = np.nanmean(prev[min_idx, :])
            if not np.isnan(avg_min):
                break
        
        for max_idx in descending_row:
            avg_max = np.nanmean(prev[max_idx, :])
            if not np.isnan(avg_max):
                break

        a = (avg_max - avg_min) / (current[max_idx] - current[min_idx])
        b = avg_max - a*current[max_idx]
        
        new[:, order[i]] = a * current + b
        np.savetxt('test.csv', new, delimiter=',')
    return new

def borda(rankings_arr, ties_treatment=1, normalization=1):
    row, col = rankings_arr.shape
    count = np.empty((row, col))
    count[:] = np.nan
    for i in range(row):
        for j in range(col):
            if not np.isnan(rankings_arr[i, j]):
                value = rankings_arr[i, j]
                count[i, j] = np.nansum(rankings_arr[:, j] > value)
                if ties_treatment == 1:
                    count[i, j] += ((np.nansum(rankings_arr[:, j] == value)-1)/2)


    if normalization == 1:
        ranking_totals = np.nansum(count, axis=0)
        ranking_valids = len(count)
        np.nan_to_num(count, copy=False)
        max_score = max(ranking_totals)
        surplus = max_score - ranking_totals
        additional = surplus / ranking_valids
        
        count = count + additional

    if normalization == 2:
        count = rescale(count)


    print(count.sum(axis=0))
    score = np.nansum(count ,axis=1)
    print(score)
    return ss.rankdata(-score, method='min')

if __name__ == '__main__':
    rankings = pd.read_pickle('compiled_main.pkl')

    rankings = rankings.sort_values('PR_rank').iloc[:, :-1]
    print(rankings)
    rankings_arr = rankings.to_numpy()
    print(rankings_arr)
    agg = (borda(rankings_arr, normalization=0))
    agg2 = (borda(rankings_arr, normalization=1))
    print(confirm_kemeny(rankings_arr, agg))
    print(confirm_kemeny(rankings_arr, agg2))