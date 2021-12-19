import numpy as np

# y_nearest = sorted(np.sum(similarity_list[:-1], axis=1))[:k]


def vote(y_nearest, policy='most_voted', mp_k=1):
    votes = [y[-1] for y in y_nearest]

    if policy == 'most_voted':
        return max(votes, key = votes.count)

    if policy == 'mod_plurality':
        unique, counts = np.unique(votes, return_counts=True)
        # Sort unique values by their frequency
        counts_sort_ind = np.argsort(-counts)
        unique = unique[counts_sort_ind]
        counts = counts[counts_sort_ind]

        if len(counts) > 1 and counts[0] == counts[1]:
            return vote(y_nearest[:mp_k], policy='mod_plurality', mp_k=mp_k)
        else:
            return unique[0]


