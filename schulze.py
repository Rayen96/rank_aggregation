import pandas as pd
import numpy as np



class LinkStrength:
    def __init__(self, first, second):
        self.first = first
        self.second = second
        
    def __gt__(self, other):
        return self.first - self.second > other.first - other.second


    def __lt__(self, other):
        return self.first - self.second < other.first - other.second


    def __eq__(self, other):
        return self.first - self.second == other.first - other.second


    def __repr__(self):
        return (f'({self.first}, {self.second})')
class Schulze:
    def __init__(self, rankings_df, num_voters):
        self.rankings_df = rankings_df
        self.C = len(rankings_df)
        self.num_voters = num_voters
    
    def create_pairs(self):
        self.rankings_df['id'] = 1
        cart = pd.merge(self.rankings_df, self.rankings_df, on='id')
        cart = cart[cart['name_x'] != cart['name_y']]
        scores_A = cart.iloc[:, 1:1+self.num_voters].to_numpy()
        scores_B = cart.iloc[:, self.num_voters+3:2*self.num_voters+3].to_numpy()
        
        num_win = (scores_A < scores_B).sum(axis=1)
        
        
        win_margin = num_win
        cart['win_margin'] = win_margin
        
        N = cart.pivot(index='name_x', columns='name_y', values='win_margin')
        self.N = N.to_numpy()

    def _algorithm_initialization(self):
        self.winner = np.zeros(self.C, bool)
        self.strength = np.empty((self.C, self.C), dtype=object)
        self.strength.fill(np.nan)
        self.pred = np.empty((self.C, self.C), dtype=object)
        self.pred.fill(np.nan)
        self.relation_O = set()
        
        for i in range(self.C):
            for j in range(self.C):
                if (i != j):
                    self.strength[i, j] = LinkStrength(self.N[i, j], self.N[j, i])
                    self.pred[i, j] = i
                    
                    
    def _strongest_path(self):
        for i in range(self.C):
            for j in range(self.C):
                if (i != j):
                    for k in range(self.C):
                        if (i != k) and (j != k):
                            if self.strength[j, k] < min(self.strength[j, i], self.strength[i, k]):
                                self.strength[j, k] = min(self.strength[j, i], self.strength[i, k])
                                self.pred[j, k] = self.pred[i, k]
        
    def _calculate_relation(self):
        for i in range(self.C):
            self.winner[i] = True
            for j in range(self.C):
                if (i != j):
                    if self.strength[j, i] > self.strength[i, j]:
                        
                        self.relation_O.add((j, i))
                        self.winner[i] = False
        
    def floyd_warshall(self):
        self.create_pairs()
        self._algorithm_initialization()
        self._strongest_path()
        self._calculate_relation()
        
'''   
rankings = pd.read_pickle('compiled_main.pkl')
rankings = rankings.sort_values('PR_rank')[:100]
rankings = rankings[rankings.isna().sum(axis=1)<8].reset_index()

schulze = Schulze(rankings)
schulze.floyd_warshall()
'''
rankings = pd.read_csv('test/example_10.csv', encoding='utf-8-sig')
schulze = Schulze(rankings, 138)
schulze.floyd_warshall()