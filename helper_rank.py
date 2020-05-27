from majority_vote import majority_vote_func
import pandas as pd
import numpy as np
from scipy import stats
from collections import Counter
from statistics import mean

def helper_rank_func(rank_function, data, methods_columns_name, num_feats):
    # calculate rank correlation and p-value. null hypothesis is that two sets of data are uncorrelated
    all_corr = np.zeros((data.shape[1], data.shape[1]))
    all_p_value = np.zeros((data.shape[1], data.shape[1]))
    for i in range(data.shape[1]):
        for j in range(data.shape[1]):
            corr, p_value = rank_function(data[:, i], data[:, j])
            all_corr[i, j] = corr
            all_p_value[i, j] = p_value

    # store correlation and p-value in dataframe
    df_corr = pd.DataFrame(all_corr, columns=methods_columns_name, index=methods_columns_name)
    df_p_value = pd.DataFrame(all_p_value, columns=methods_columns_name, index=methods_columns_name)

    # generate truth table for p-value. If p value < 0.05(significant) value is True/1.0
    df_p_value_significance = df_p_value < 0.05

    # remove diagonal significance as it is measured against itself
    for i in range(df_p_value.shape[0]):
        df_p_value_significance.iloc[i, i] = None

    # decide if feature importance method should be kept
    # if majority are correlated and statistically significant then method are kept
    keep_FI_method = []
    non_significant_majority_counter = 0
    keep_or_not = True

    # zeros placeholder same number of rows/feature
    all_stacked_corr = np.zeros((data.shape[0], 1))
    # loop to include only majority voted method. Majority voted in this case is equivalent
    # to all method deemed correlated by statistical significance
    for i in range(df_p_value_significance.shape[1]):
        keep_or_not = Counter(df_p_value_significance.iloc[:, i]).most_common()[0][0]
        keep_FI_method.append(keep_or_not)
        if keep_or_not:
            all_stacked_corr = np.hstack((all_stacked_corr, np.reshape(data[:, i], (-1, 1))))
        else:
            non_significant_majority_counter =+ 1

    # if majority of feature importance method do not correlate skip to majority voting
    if ((non_significant_majority_counter/df_p_value_significance.shape[1])>0.5):
        # remove the zeros placeholder for shape
        all_stacked_corr = all_stacked_corr[:, 1:]
        # calculate mean for each feature
        total_scaled_corr = np.mean(all_stacked_corr, axis=1)
        return total_scaled_corr, df_p_value
    else:
        total_scaled_corr = majority_vote_func(data, num_feats)
        return total_scaled_corr, df_p_value