import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.spatial.distance import pdist, squareform
from scipy.stats import f
import csv

CLASSES = set()
SIGMA = np.array([])
COUNTS = {}
TRAIN_DATA = pd.DataFrame()


def init_hvdm(x_num, x_cat: DataFrame, labels):
    global SIGMA
    global CLASSES
    global COUNTS
    global TRAIN_DATA

    TRAIN_DATA = x_cat
    TRAIN_DATA['class'] = labels
    SIGMA = np.std(x_num, axis=0)
    CLASSES = set(labels)
    for i, att in enumerate(x_cat.keys()):
        att_class = x_cat[att].to_frame()
        att_class['class'] = labels
        COUNTS[att] = pd.DataFrame(att_class.value_counts(sort=False, dropna=False).reset_index().values)
        columns = att_class.columns.to_list()
        columns.append('Count')
        COUNTS[att].columns = columns


def hvdm(x_num, x_cat: DataFrame, y_num, y_cat: DataFrame):
    """
    Computes the heterogeneous value difference between two samples.
    :param x_num: numerical values of x
    :param y_num: numerical values of y
    :param x_cat: categorical values of x.
    :param y_cat: categorical values of y.
    :return: distance between x and y
    """

    if not CLASSES:
        raise ValueError("Please initialize hvdm parameter running init_hvdm before using hvdm.")

    het_dist = 0

    het_dist += np.nansum(np.square(abs(x_num - y_num) / (4 * SIGMA)))

    for i, att in enumerate(x_cat.keys()):
        if x_cat.iloc[0, i] == '?' or y_cat.iloc[0, i] == '?':
            het_dist += 1
        else:
            for c in CLASSES:
                aux_df = COUNTS[att]
                n_axc = aux_df[(aux_df[att] == x_cat.iloc[0, i]) & (aux_df['class'] == c)]['Count'].to_list()
                if not n_axc:
                    p_axc = 0
                else:
                    n_ax = aux_df[aux_df[att] == x_cat.iloc[0, i]]['Count'].sum()
                    p_axc = n_axc[0] / n_ax

                n_ayc = aux_df[(aux_df[att] == y_cat.iloc[0, i]) & (aux_df['class'] == c)]['Count'].to_list()
                if not n_ayc:
                    p_ayc = 0
                else:
                    n_ay = aux_df[aux_df[att] == y_cat.iloc[0, i]]['Count'].sum()
                    p_ayc = n_ayc[0] / n_ay

                het_dist += (p_axc - p_ayc) ** 2

    return np.sqrt(het_dist)


def hvdm_v2(x_num, x_cat: DataFrame, y_num, y_cat: DataFrame):
    """
    Computes the heterogeneous value difference between two samples.
    :param x_num: numerical values of x
    :param y_num: numerical values of y
    :param x_cat: categorical values of x.
    :param y_cat: categorical values of y.
    :return: distance between x and y
    """

    if not CLASSES:
        raise ValueError("Please initialize hvdm parameter running init_hvdm before using hvdm.")

    het_dist = 0

    het_dist += np.nansum(np.square(abs(x_num - y_num) / (4 * SIGMA)))

    for i, att in enumerate(x_cat.keys()):
        if x_cat.iloc[0, i] == '?' or y_cat.iloc[0, i] == '?':
            het_dist += 1
        else:
            n_axc = TRAIN_DATA[[att, 'class']][(TRAIN_DATA[att] == x_cat.iloc[0, i])].groupby('class').count()[att]
            n_ax = n_axc.sum().item()

            p_axc = n_axc / n_ax

            n_ayc = TRAIN_DATA[[att, 'class']][(TRAIN_DATA[att] == y_cat.iloc[0, i])].groupby('class').count()[att]
            n_ay = n_ayc.sum().item()

            p_ayc = n_ayc / n_ay

            het_dist += ((p_axc.sub(p_ayc, fill_value=0))**2).sum()


def cat_diff(x, y):
    result = []
    for i in range(len(x)):
        if x[i] == '?' and y[i] == '?':
            result.append(1)
        else:
            result.append(x[i] != y[i])
    return result


def cat_sum(x, y):
    result = []
    for i in range(len(x)):
        if x[i] == '?' and y[i] == '?':
            result.append(1)
        else:
            result.append(1 + (x[i] == y[i]))
    return result


def cat_prod(x, y):
    result = []
    for i in range(len(x)):
        if x[i] == '?' and y[i] == '?':
            result.append(0)
        else:
            result.append(x[i] == y[i])
    return result


def num_distance(x, y, metric='euclidean'):
    if metric == 'euclidean':
        return np.sqrt(np.square(x - y).sum())

    if metric == 'manhattan':
        return abs(x - y).sum()

    if metric == 'cosine':
        sim = (x * y).sum() / (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
        return 1 - sim

    if metric == 'clark':
        return np.nansum(np.square(x - y) / np.square(x + y))

    if metric == 'canberra':
        return np.nansum(abs(x - y) / abs(x + y))


def distance(x_num, y_num, x_cat=None, y_cat=None, metric='euclidean'):
    """
    Calculates distance between two samples according to some metric.
    :param x_num: numerical features of the x samples
    :param y_num: numerical features of the y samples
    :param x_cat: categorical features of the x samples
    :param y_cat: categorical features of the y samples
    :param metric: distance metric
    :return: distance between x and y
    """

    if x_cat is None or y_cat is None:
        return num_distance(x_num, y_num, metric)

    # Distance for categorical attributes
    if metric == 'hamming':
        dist = 0
        for i in range(len(x_cat)):
            if x_cat[i] is None or y_cat[i] is None:
                dist += 1
            else:
                dist += (x_cat[i] != y_cat[i])
        return dist

    if metric == 'euclidean':
        return np.sqrt(np.square(np.concatenate((x_num - y_num, cat_diff(x_cat, y_cat)))).sum())

    if metric == 'manhattan':
        return abs(np.concatenate((x_num - y_num, cat_diff(x_cat, y_cat)))).sum()

    if metric == 'cosine':
        sim = (np.concatenate((x_num * y_num, cat_prod(x_cat, y_cat)))).sum() \
              / (np.sqrt((np.concatenate((x_num * x_num, cat_prod(x_cat, x_cat)))).sum())
                 * np.sqrt((np.concatenate((y_num * y_num, cat_prod(y_cat, y_cat)))).sum()))
        return 1 - sim

    # In the Clark and Canberra measures we use np.nansum since there might be divisions by zero.
    if metric == 'clark':
        return np.nansum(np.square(np.concatenate((x_num - y_num, cat_diff(x_cat, y_cat)))) / np.square(
            np.concatenate((x_num + y_num, cat_sum(x_cat, y_cat)))))

    if metric == 'canberra':
        numerator = abs(np.concatenate((x_num - y_num, cat_diff(x_cat, y_cat))))
        denominator = abs(np.concatenate((x_num + y_num, cat_sum(x_cat, y_cat))))
        return np.nansum(numerator / denominator)


def vote(votes, policy='most_voted', mp_k=1):
    """
    Computes the class of a sample according to
    the classes of its k neighbours

    :param votes: classes of the k nearest neighbours
    :param policy: Most Voted Solution, Modified Plurality or Borda Count
    :param mp_k: number of neighbours to remove in case of tie in the Modified Plurality policy
    :return: winner class
    """

    if policy == 'most_voted':
        return max(votes, key=votes.count)

    if policy == 'mod_plurality':
        # Unique options sorted by decreasing number of votes
        options_srt = sorted(set(votes), key=votes.count, reverse=True)
        # Votes for each option
        count_srt = [votes.count(x) for x in options_srt]

        # In case of tie, remove mp_k neighbours and repeat
        if len(options_srt) > 1:
            if count_srt[0] == count_srt[1]:
                if len(votes) <= mp_k:
                    mp_k = 1
                return vote(votes[:-mp_k], policy='mod_plurality', mp_k=mp_k)
        else:
            return options_srt[0]

    if policy == 'borda_count':
        # Dictionary of unique options
        options = dict.fromkeys(set(votes), 0)

        # Points for each element in votes
        points = list(range(len(votes)))[::-1]
        # Assign total points to each option in the dictionary
        for opt in options.keys():
            opt_points = [points[i] for i in np.where(np.array(votes) == opt)[0]]
            options[opt] = sum(opt_points)

        return max(options, key=options.get)


def friedman_nemenyi(groups, alpha=0.05):
    """
    Friedman test followed by the Nemenyi post-hoc test.
    It finds whether k algorithms are significantly different
    based on n measures for each of them.
    :param groups: k x n Numpy array of measures
    :param alpha: significance level for the critical values
    :return:
    """
    k, n = groups.shape
    ranks = groups.argsort(axis=0)
    ranks_mean = ranks.mean(axis=1)

    xi_square = (12 * n / (k * (k + 1))) * (sum(ranks_mean ** 2) - ((k * (k + 1) ** 2) / 4))
    ff = ((n - 1) * xi_square) / (n * (k - 1) - xi_square)
    crit_val = f.ppf(q=alpha, dfn=k - 1, dfd=(k - 1) * (n - 1))

    if ff > crit_val:
        pair_diff = pdist(ranks_mean[:, None], metric='minkowski')
        diff_matrix = squareform(pair_diff)
        crit_dist = cd_nememyi(alpha)[str(k)]
        which_diff = diff_matrix > crit_dist
        return ff, crit_val, which_diff, crit_dist

    else:
        return ff, crit_val, False, None


def cd_nememyi(alpha):
    with open('cd_nemenyi.csv', mode='r') as file:
        reader = csv.DictReader(file)
        dict_from_csv = {rows['models']: rows[f'Nemenyi {alpha}]'] for rows in reader}
    return dict_from_csv
