import numpy as np
from data_reader import read_df
import networkx as nx
import pandas as pd
import scipy.stats as ss

def generate_transition_matrix(rankings):
    ranked_higher = (rankings[:, None, None] < rankings[None, :, None]).sum(axis=-1).reshape(rankings.shape[0], rankings.shape[0])
    valid_ranked = (~np.isnan(rankings[:, None, None]) * ~np.isnan(rankings[None, :, None])).sum(axis=-1).reshape(rankings.shape[0], rankings.shape[0])
    return ((ranked_higher / valid_ranked) >= 0.5).astype(int)


def MC4(rankings_df):
    rankings = rankings_df.to_numpy()

    transition_matrix = generate_transition_matrix(rankings)

    G = nx.from_numpy_matrix(transition_matrix, create_using = nx.DiGraph)
    node_mapping = {i: rankings_df.index[i] for i in range(len(rankings_df))}
    G = nx.relabel_nodes(G, node_mapping)
    PR_res = nx.pagerank(G)
    PR_df = pd.DataFrame.from_dict(PR_res, orient='index')
    
    PR_df['PR_rank'] = ss.rankdata(PR_df[0], method='min')

    return PR_df

if __name__ == '__main__':
    rankings_df = read_df('../merged_2021-01-19.csv')
    print(MC4(rankings_df))

