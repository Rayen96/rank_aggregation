import pandas as pd

def read_df(file_name):
    data = pd.read_csv(file_name)
    print(len(data))
    data = data[data[['ARWU', 'THE', 'USNWR', 'QS', 'CUG', 'BCNR', 'Forbes', 'MHRD']].isna().sum(axis=1) < 8]
    print(len(data))
    data = data.set_index('name')

    ranking_df = data[['ARWU', 'THE', 'USNWR', 'QS', 'CUG', 'BCNR', 'Forbes', 'MHRD']]
    return ranking_df

if __name__ == '__main__':
    df = read_df('../merged_2021-01-19.csv')

    print(df)