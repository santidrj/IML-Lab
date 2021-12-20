import numpy as np
import pandas as pd

"""
----> self.sigma <---- = numerical_features.std().to_list()
----> self.labels <---- = self.training_set.iloc[:, -1]
numerical_features, ----> self.cat_features <---- = preprocess(self.training_set)
"""
cat_features = []   # Pandas Dataframe
labels = []
sigma = []


def hvdm(x_num, x_cat, y_num, y_cat):
    """
    Computes the heterogeneous value difference between two samples.
    :param x_num, y_num: numerical values of x and y
    :param x_cat, y_cat: categorical values of x and y.
    :return: distance between x and y
    """

    het_dist = 0

    for i in range(len(x_num)):
        if x_num[i] is None or y_num[i] is None:
            het_dist += 1
        else:
            het_dist += np.square(abs(x_num[i] - y_num[i]) / (4 * sigma[i]))

    for i, att in enumerate(x_cat.keys()):
        if x_cat[i] is None or y_cat[i] is None:
            het_dist += 1
        else:
            att_class = cat_features[att]
            att_class['class'] = labels
            counts = pd.DataFrame(att_class.value_counts(sort=False, dropna=False).reset_index().values)
            counts.columns = att_class.columns.to_list().append('Count')

            for c in att_class['class'].unique():

                n_axc = counts['Count'][counts[att] == x_cat[i]][counts['class'] == c].item()
                if n_axc.empty:
                    p_axc = 0
                else:
                    n_ax = counts['Count'][counts[att] == x_cat[i]].sum()
                    p_axc = n_axc / n_ax

                n_ayc = counts['Count'][counts[att] == y_cat[i]][counts['class'] == c].item()
                if n_ayc.empty:
                    p_ayc = 0
                else:
                    n_ay = counts['Count'][counts[att] == x_cat[i]].sum()
                    p_ayc = n_ayc / n_ay

                het_dist += (p_axc - p_ayc)**2

    return np.sqrt(het_dist)


def distance(x, y, metric):
    """
    Calculates distance between two samples according to some metric.
    :param x, y: samples
    :param metric: distance metric
    :return: distance between x and y
    """

    # Distance for categorical attributes
    if metric == 'hamming':
        dist = 0
        for i in range(len(x)):
            if x[i] is None or y[i] is None:
                dist += 1
            else:
                dist += (x[i] != y[i])
        return dist

    if metric == 'euclidean':
        return np.sqrt(np.square(x - y).sum())

    if metric == 'manhattan':
        return abs(x - y).sum()

    if metric == 'cosine':
        sim = (x * y).sum() / (np.sqrt((x * x).sum()) * np.sqrt((y * y).sum()))
        return 1 - sim

    if metric == 'clark':
        return (np.square(x - y)/np.square(x + y)).sum()

    if metric == 'canberra':
        return (abs(x - y)/abs(x + y)).sum()


def vote(votes, policy='most_voted', mp_k=1):
    """
    Computes the class of a sample according to
    the classes of its k neighbours

    :param votes: classes of the k nearest neighbours
        ---REMOVE---
        y_nearest = sorted(np.sum(similarity_list[:-1], axis=1))[:k]
        votes = [y[-1] for y in y_nearest]
        ---REMOVE---
    :param policy: Most Voted Solution, Modified Plurality or Borda Count
    :param mp_k: number of neighbours to remove in case of tie in the Modified Plurality policy
    :return: winner class
    """

    if policy == 'most_voted':
        """
        ---REMOVE---
        Note to my dear partners:
        In case of tie among classes, this policy returns the first occurrence 
        of those classes in "votes", which corresponds to the nearest 
        neighbour of that class because "votes" is sorted according to distance to
        the sample
        ---REMOVE---
        """
        return max(votes, key=votes.count)

    if policy == 'mod_plurality':

        # Unique options sorted by decreasing number of votes
        options_srt = sorted(set(votes), key=votes.count, reverse=True)
        # Votes for each option
        count_srt = [votes.count(x) for x in options_srt]

        # In case of tie, remove mp_k neighbours and repeat
        if len(votes) > 1 and count_srt[0] == count_srt[1]:
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

        """
        ---REMOVE---
        Same reasoning as "most_voted" in case of tie
        ---REMOVE---
        """
        return max(options, key=options.get)







