import pandas as pd
import numpy as np
import scipy.stats as ss


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
    return new

if __name__ == '__main__':
    rankings = pd.read_pickle('compiled_main.pkl')
    rankings = rankings.sort_values('PR_rank').iloc[:, :-1]
    rankings_arr = rankings.to_numpy()
    print(rankings_arr)
    rescaled = rescale(rankings_arr)
    print(rescaled)
    