import numpy as np


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
        if len(count_srt) > 1 and count_srt[0] == count_srt[1]:
            return vote(votes[:mp_k], policy='mod_plurality', mp_k=mp_k)
        else:
            return options_srt[0]





