from numpy.ma import masked, nomask
from scipy.stats._stats_mstats_common import _find_repeats
import itertools
import scipy.special as special
import copy
from numpy import ma
import numpy as np
from data_reader import read_df
import scipy.stats as ss

def _chk_size(a, b):
    a = ma.asanyarray(a)
    b = ma.asanyarray(b)
    (na, nb) = (a.size, b.size)
    if na != nb:
        raise ValueError("The size of the input array should match!"
                         " (%s <> %s)" % (na, nb))
    return (a, b, na)


def rankdata(data, axis=None, use_missing=False):
    """Returns the rank (also known as order statistics) of each data point
    along the given axis.
    If some values are tied, their rank is averaged.
    If some values are masked, their rank is set to 0 if use_missing is False,
    or set to the average rank of the unmasked values if use_missing is True.
    Parameters
    ----------
    data : sequence
        Input data. The data is transformed to a masked array
    axis : {None,int}, optional
        Axis along which to perform the ranking.
        If None, the array is first flattened. An exception is raised if
        the axis is specified for arrays with a dimension larger than 2
    use_missing : bool, optional
        Whether the masked values have a rank of 0 (False) or equal to the
        average rank of the unmasked values (True).
    """
    def _rank1d(data, use_missing=False):
        n = data.count()
        rk = np.empty(data.size, dtype=float)
        idx = data.argsort()
        rk[idx[:n]] = np.arange(1,n+1)

        if use_missing:
            rk[idx[n:]] = (n+1)/2.
        else:
            rk[idx[n:]] = 0

        repeats = find_repeats(data.copy())
        for r in repeats[0]:
            condition = (data == r).filled(False)
            rk[condition] = rk[condition].mean()
        return rk

    data = ma.array(data, copy=False)
    if axis is None:
        if data.ndim > 1:
            return _rank1d(data.ravel(), use_missing).reshape(data.shape)
        else:
            return _rank1d(data, use_missing)
    else:
        return ma.apply_along_axis(_rank1d,axis,data,use_missing).view(ndarray)


def find_repeats(arr):
    """Find repeats in arr and return a tuple (repeats, repeat_count).
    The input is cast to float64. Masked values are discarded.
    Parameters
    ----------
    arr : sequence
        Input array. The array is flattened if it is not 1D.
    Returns
    -------
    repeats : ndarray
        Array of repeated values.
    counts : ndarray
        Array of counts.
    """
    # Make sure we get a copy. ma.compressed promises a "new array", but can
    # actually return a reference.
    compr = np.asarray(ma.compressed(arr), dtype=np.float64)
    try:
        need_copy = np.may_share_memory(compr, arr)
    except AttributeError:
        # numpy < 1.8.2 bug: np.may_share_memory([], []) raises,
        # while in numpy 1.8.2 and above it just (correctly) returns False.
        need_copy = False
    if need_copy:
        compr = compr.copy()
    return _find_repeats(compr)

    
def count_tied_groups(x, use_missing=False):
    """
    Counts the number of tied values.
    Parameters
    ----------
    x : sequence
        Sequence of data on which to counts the ties
    use_missing : bool, optional
        Whether to consider missing values as tied.
    Returns
    -------
    count_tied_groups : dict
        Returns a dictionary (nb of ties: nb of groups).
    Examples
    --------
    >>> from scipy.stats import mstats
    >>> z = [0, 0, 0, 2, 2, 2, 3, 3, 4, 5, 6]
    >>> mstats.count_tied_groups(z)
    {2: 1, 3: 2}
    In the above example, the ties were 0 (3x), 2 (3x) and 3 (2x).
    >>> z = np.ma.array([0, 0, 1, 2, 2, 2, 3, 3, 4, 5, 6])
    >>> mstats.count_tied_groups(z)
    {2: 2, 3: 1}
    >>> z[[1,-1]] = np.ma.masked
    >>> mstats.count_tied_groups(z, use_missing=True)
    {2: 2, 3: 1}
    """
    nmasked = ma.getmask(x).sum()
    # We need the copy as find_repeats will overwrite the initial data
    data = ma.compressed(x).copy()
    (ties, counts) = find_repeats(data)
    nties = {}
    if len(ties):
        nties = dict(zip(np.unique(counts), itertools.repeat(1)))
        nties.update(dict(zip(*find_repeats(counts))))

    if nmasked and use_missing:
        try:
            nties[nmasked] += 1
        except KeyError:
            nties[nmasked] = 1

    return nties

    
def kendalltau(x, y, use_ties=True, use_missing=False, method='auto'):
    """
    Computes Kendall's rank correlation tau on two variables *x* and *y*.
    Parameters
    ----------
    x : sequence
        First data list (for example, time).
    y : sequence
        Second data list.
    use_ties : {True, False}, optional
        Whether ties correction should be performed.
    use_missing : {False, True}, optional
        Whether missing data should be allocated a rank of 0 (False) or the
        average rank (True)
    method: {'auto', 'asymptotic', 'exact'}, optional
        Defines which method is used to calculate the p-value [1]_.
        'asymptotic' uses a normal approximation valid for large samples.
        'exact' computes the exact p-value, but can only be used if no ties
        are present. As the sample size increases, the 'exact' computation
        time may grow and the result may lose some precision.
        'auto' is the default and selects the appropriate
        method based on a trade-off between speed and accuracy.
    Returns
    -------
    correlation : float
        Kendall tau
    pvalue : float
        Approximate 2-side p-value.
    References
    ----------
    .. [1] Maurice G. Kendall, "Rank Correlation Methods" (4th Edition),
           Charles Griffin & Co., 1970.
    """
    x = ma.masked_invalid(x)
    y = ma.masked_invalid(y)
    (x, y, n) = _chk_size(x, y)
    (x, y) = (x.flatten(), y.flatten())
    m = ma.mask_or(ma.getmask(x), ma.getmask(y))
    if m is not nomask:
        x = ma.array(x, mask=m, copy=True)
        y = ma.array(y, mask=m, copy=True)
        # need int() here, otherwise numpy defaults to 32 bit
        # integer on all Windows architectures, causing overflow.
        # int() will keep it infinite precision.
        n -= int(m.sum())

    if n < 2:
        return np.nan

    rx = ma.masked_equal(rankdata(x, use_missing=use_missing), 0)
    ry = ma.masked_equal(rankdata(y, use_missing=use_missing), 0)
    idx = rx.argsort()
    (rx, ry) = (rx[idx], ry[idx])
    C = np.sum([((ry[i+1:] > ry[i]) * (rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)
    D = np.sum([((ry[i+1:] < ry[i])*(rx[i+1:] > rx[i])).filled(0).sum()
                for i in range(len(ry)-1)], dtype=float)

    xties = count_tied_groups(x)
    yties = count_tied_groups(y)
    
    if use_ties:
        corr_x = np.sum([v*k*(k-1) for (k,v) in xties.items()], dtype=float)
        corr_y = np.sum([v*k*(k-1) for (k,v) in yties.items()], dtype=float)
        denom = ma.sqrt((n*(n-1)-corr_x)/2. * (n*(n-1)-corr_y)/2.)
    else:
        denom = n*(n-1)/2.
    tau = (C-D) / denom
    res = {'C': C, 'D': D, 'xties': xties, 'yties': yties, 'n': n}
    return res


def calculate_tau(res):
    n = res['n']
    corr_x = np.sum([v*k*(k-1) for (k,v) in res['xties'].items()], dtype=float)
    corr_y = np.sum([v*k*(k-1) for (k,v) in res['yties'].items()], dtype=float)
    denom = ma.sqrt((n*(n-1)-corr_x)/2. * (n*(n-1)-corr_y)/2.)
    tau = (res['C']-res['D']) / denom
    return tau


def decrement_val(count_dict, key):
    if key >= 2: 
        count_dict[key] -= 1
        if count_dict[key] == 0:
            del count_dict[key]
        

def increment_val(count_dict, key):
    if key >= 2: 
        if key in count_dict:
            count_dict[key] += 1
        else:
            count_dict[key] = 1

def update_count(aggregate, count_dict, pos1, old1, new1, pos2=None, old2=None, new2=None):
    new_aggregate = aggregate.copy()
    old_original_count_1 = (aggregate == old1).sum()
    new_original_count_1 = (aggregate == new1).sum()
    
    decrement_val(count_dict, old_original_count_1)
    decrement_val(count_dict, new_original_count_1)
    
    if old2 is not None:
        old_original_count_2 = (aggregate == old2).sum()
        new_original_count_2 = (aggregate == new2).sum()

    new_aggregate[pos1] = new1
    
    if old2 is not None:
        new_aggregate[pos2] = new2
        
    old_updated_count_1 = (new_aggregate == old1).sum()
    new_updated_count_1 = (new_aggregate == new1).sum()
    
    
    if old2 is not None:
        old_updated_count_2 = (new_aggregate == old2).sum()
        new_updated_count_2 = (new_aggregate == new2).sum()
    
    increment_val(count_dict, old_updated_count_1)
    increment_val(count_dict, new_updated_count_1)
    
    
            
def update_results(ranking, aggregate, res, pos1, rank1, pos2=None, rank2=None):
    if pos1 == pos2 or rank1 == rank2:
        raise ValueError
    
    new_aggregate = copy.deepcopy(aggregate)
    new_res = copy.deepcopy(res)
    count_dict = new_res['yties']
    
    new_res['C'] -= sum([(ranking[i] > ranking[pos1]) * (new_aggregate[i] > new_aggregate[pos1]) + (ranking[i] < ranking[pos1]) * (new_aggregate[i] < new_aggregate[pos1]) for i in range(len(aggregate))])
    new_res['D'] -= sum([(ranking[i] < ranking[pos1]) * (new_aggregate[i] > new_aggregate[pos1]) + (ranking[i] > ranking[pos1]) * (new_aggregate[i] < new_aggregate[pos1]) for i in range(len(aggregate))])
    
    
    if pos2 is not None:
        new_res['C'] -= sum([(ranking[i] > ranking[pos2]) * (new_aggregate[i] > new_aggregate[pos2]) + (ranking[i] < ranking[pos2]) * (new_aggregate[i] < new_aggregate[pos2]) for i in range(len(aggregate))])
        new_res['D'] -= sum([(ranking[i] < ranking[pos2]) * (new_aggregate[i] > new_aggregate[pos2]) + (ranking[i] > ranking[pos2]) * (new_aggregate[i] < new_aggregate[pos2]) for i in range(len(aggregate))])
    
    
    # update_count(aggregate, count_dict, pos1, aggregate[pos1], rank1, pos2=None, old2=None, new2=None)
    
    new_aggregate[pos1] = rank1
    
    if pos2 is not None:
        new_aggregate[pos2] = rank2
        
    
    new_res['C'] += sum([(ranking[i] > ranking[pos1]) * (new_aggregate[i] > new_aggregate[pos1]) + (ranking[i] < ranking[pos1]) * (new_aggregate[i] < new_aggregate[pos1]) for i in range(len(aggregate))])
    new_res['D'] += sum([(ranking[i] < ranking[pos1]) * (new_aggregate[i] > new_aggregate[pos1]) + (ranking[i] > ranking[pos1]) * (new_aggregate[i] < new_aggregate[pos1]) for i in range(len(aggregate))])
    
    
    
    if pos2 is not None:
        new_res['C'] += sum([(ranking[i] > ranking[pos2]) * (new_aggregate[i] > new_aggregate[pos2]) + (ranking[i] < ranking[pos2]) * (new_aggregate[i] < new_aggregate[pos2]) for i in range(len(aggregate))])
        new_res['D'] += sum([(ranking[i] < ranking[pos2]) * (new_aggregate[i] > new_aggregate[pos2]) + (ranking[i] > ranking[pos2]) * (new_aggregate[i] < new_aggregate[pos2]) for i in range(len(aggregate))])
    valid = new_aggregate[~np.isnan(ranking)]
    new_res['yties'] = count_tied_groups(valid)
        
    return new_res


def generate_score_list(rankings_arr, aggregate):
    results = []
    for i in range(rankings_arr.shape[1]):
        results.append(kendalltau(rankings_arr[:, i], aggregate))
    return results


def calculate_kemeny_distance(score_list):
    return sum([calculate_tau(x) for x in score_list])


def update_score_list(rankings_arr, aggregate, score_list, pos, rank):
    for i in range(len(score_list)):
        score_list[i] = update_results(rankings_arr[:, i], aggregate, score_list[i], pos, rank)


def confirm_kemeny(rankings_arr, aggregate):
    return sum([ss.kendalltau(rankings_arr[:, i], aggregate, nan_policy='omit').correlation for i in range(rankings_arr.shape[1])])


if __name__ == '__main__':
    rankings_df = read_df('../merged_2021-01-19.csv')
    rankings_arr = rankings_df.to_numpy()

    aggregate = np.nanmedian(rankings_arr, axis=1)

    score_list = generate_score_list(rankings_arr, aggregate)
    kemeny1 = calculate_kemeny_distance(score_list)
    kemeny2 = confirm_kemeny(rankings_arr, aggregate)

    print(kemeny1 == kemeny2)

    update_score_list(rankings_arr, aggregate, score_list, 150, 1)

    kemeny3 = calculate_kemeny_distance(score_list)
    aggregate[150] = 1
    kemeny4 = confirm_kemeny(rankings_arr, aggregate)

    print(kemeny3 == kemeny4)