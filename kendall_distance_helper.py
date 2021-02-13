import scipy.stats as ss
import numpy as np

def score(i1, i2, a1, a2):
    if not np.isnan(i1) and not np.isnan(i2):
        if (i1 < i2 and a1 > a2) or (i1 > i2 and a1 < a2):
            return 1
        elif (i1 != i2 and a1 == a2) or (i1 == i2 and a1 != a2):
            return 1
        else:
            return 0
    else:
        return 0


def kendall_tau_distance(input_ranking, aggregate, pair_update = None):
    N = len(aggregate)
    dist = 0
    
    for i in range(N):        
        if pair_update is None:
            for j in range(i):
                dist += score(input_ranking[i], input_ranking[j], aggregate[i], aggregate[j])
        else:
            if i != pair_update:
                dist += score(input_ranking[i], input_ranking[pair_update], aggregate[i], aggregate[pair_update])
            
    return dist



def kemeny_distance(input_ranking_matrix, aggregate, pair_update = None):
    N, D = input_ranking_matrix.shape
    
    return sum(kendall_tau_distance(input_ranking_matrix[:, i], aggregate, pair_update) for i in range(D))

if __name__ == '__main__':
    input_rankings = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [3, 2, 2, 3],
        [4, 4, 4, 4]
    ])
    aggregate = np.array([1, 2, 2, 4])
    print(kemeny_distance(input_rankings, aggregate))