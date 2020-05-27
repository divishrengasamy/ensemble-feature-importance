import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from statistics import mean
def majority_vote_func(all_stacked, num_feats):
    # obtain rank of each feature for each feature importance
    all_stacked_rank = np.zeros((num_feats, 1))
    for i in range(all_stacked.shape[1]):
        all_stacked_rank = np.hstack((all_stacked_rank, np.reshape(np.argsort(all_stacked[:, i]), (-1, 1))))

    all_stacked_rank = all_stacked_rank[:, 1:]

    # select the most common rank for each feature
    most_common_rank = []
    for i in range(all_stacked_rank.shape[0]):
        majority_number = Counter(all_stacked_rank[i, :]).most_common()[0][1]
        if len(Counter(all_stacked_rank[i, :]).most_common()) > 1:
            second_majority_number = Counter(all_stacked_rank[i, :]).most_common()[1][1]
        else:
            second_majority_number = None
        # record instances if there are two majority, else just append the majority
        if (majority_number) == (second_majority_number):
            most_common_rank.append(
                [
                    int(Counter(all_stacked_rank[i, :]).most_common()[0][0]),
                    int(Counter(all_stacked_rank[i, :]).most_common()[1][0]),
                ]
            )
        else:
            most_common_rank.append(([int(Counter(all_stacked_rank[i, :]).most_common()[0][0])]))

    # average of majorty
    total_scaled_majority_vote = list()
    for i in range(all_stacked_rank.shape[0]):
        temp_majority_avg = []
        for j in range(all_stacked_rank.shape[1]):
            # single majority
            if len(most_common_rank[i]) == 1:
                if all_stacked_rank[i, j] == most_common_rank[i][0]:
                    temp_majority_avg.append(all_stacked[i, j])
            # double majority
            else:
                if (all_stacked_rank[i, j] == most_common_rank[i][0]) or (all_stacked_rank[i, j] == most_common_rank[i][1]):
                    temp_majority_avg.append(all_stacked[i, j])
        total_scaled_majority_vote.append(mean(temp_majority_avg))
    total_scaled_majority_vote = np.array(total_scaled_majority_vote)
    return total_scaled_majority_vote