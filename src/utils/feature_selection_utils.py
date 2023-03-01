import pandas as pd
import numpy as np
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import SelectKBest
from mlxtend.feature_selection import SequentialFeatureSelector as sfs
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


def spearman_corr(df, dep_col):
    """This function is used to find the spearman correlation of columns
    to be selected as features.

    param df: dataset as dataframe

    type df: pandas dataframe

    param dep_col: dependent column name i.e output

    type dep_col: string

    return: spearman coefficient for all columns.

    rtype: pandas series
    """
    spear_coef = df.corr(method="spearman")[dep_col]
    return spear_coef


def convert_result_series_to_df(ser, colname_list):
    """This function is used to convert series to a pandas dataframe.

    param ser: series to be converted.

    type ser: pandas series

    param colname_list:list of column names for dataframe

    type colname_list: list

    return: resultant dataframe

    rtype: pandas dataframe
    """

    res_df = pd.DataFrame(ser)
    res_df.columns = colname_list
    return res_df


def chisquare_test(df, dep_col):
    """This function is used to chi-square values for the input dataframe
    for all of its columns.

    param df: dataset as dataframe

    type df: pandas dataframe

    param dep_col: dependent column name i.e output

    type dep_col: string

    return: chi-square values for all columns.

    rtype: pandas series
    """

    list_chi_score = []
    list_col_names = []
    for i in df.columns:
        X = np.array(df[i]).reshape(-1, 1)
        y = df[dep_col]
        chi_scores = chi2(X, y)
        list_chi_score.append(chi_scores[1][0])
        chi_score_series = pd.Series(v for v in list_chi_score)
        list_col_names.append(i)
        chi_score_series.index = list_col_names
        chi_score_series.sort_values(ascending=False)
    return chi_score_series


def convert_list_to_series(lst):
    """This function is used to convert list to a pandas series.

    param lst: list to be converted.

    type lst: list

    return: resultant series

    rtype: pandas series
    """
    series = pd.Series(v for v in lst)
    return series


def col_list_with_1_class(df, num_list):
    """This function is used to find column list which have only one class.

    param df: dataset as dataframe

    type df: pandas dataframe

    param num_list: list of numerical columns

    type num_list: list

    return: column list which are to be dropped.

    rtype: list
    """

    col_list_to_drop = []
    for i in num_list:
        if len(df[i].value_counts()) == 1:
            col_list_to_drop.append(i)
    return col_list_to_drop


def mutualinfo_values(xtrain, ytrain):
    """This function is used to find mutual info values for the given xtrain,
     ytrain.

    param xtrain: splitted train dataset without dependent column in the form
    of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    return: mutual info values as series.

    rtype: pandas series.
    """

    mutual_info = mutual_info_classif(xtrain, ytrain)
    mutual_info = pd.Series(mutual_info)
    mutual_info.index = xtrain.columns
    mutual_info.sort_values(ascending=False)
    return mutual_info


def woe_iv_values(xtrain, ytrain):
    """This function is used to find woe-iv values for the given xtrain,
     ytrain.

    param xtrain: splitted train dataset without dependent column in the form
    of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    return: woe-iv values as series.

    rtype: pandas series.
    """

    woe_iv_sum_list = []
    col_list = []
    for i in xtrain.columns:
        df_woe_iv = (pd.crosstab(xtrain[i], ytrain, normalize='columns')
                     .assign(woe=lambda dfx: np.log(dfx[1] / dfx[0]))
                     .assign(iv=lambda dfx: np
                     .sum(dfx['woe'] * (dfx[1] - dfx[0]))))
        # print(df_woe_iv)
        woe_iv_sum_list.append(df_woe_iv.iv.sum())
        woe_iv_sum_series = pd.Series(v for v in woe_iv_sum_list)
        col_list.append(i)
        woe_iv_sum_series.index = col_list
        woe_iv_sum_series.sort_values(ascending=False)
    return woe_iv_sum_series


def Analysis_Report(list_of_df):
    """This function is used to find merged dataframe for the given
    list of dataframes.

    param list_of_df: list of dataframes to be merged.

    type list_of_df: list

    return: merged dataframe.

    rtype: pandas dataframe.
    """

    merged_df = pd.concat((list_of_df), axis=1)
    return merged_df


def selectkbest_features_mutual_classif(xtrain, ytrain, K):
    """This function is used to find k best features using mutual info classifier
     for the given xtrain, ytrain.

    param xtrain: splitted train dataset without dependent column in the form
    of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    param K: no. of features to be selected.

    type K: int

    return: list of k best features

    rtype: list
    """

    sel_five_cols = SelectKBest(mutual_info_classif, k=K)
    sel_five_cols.fit(xtrain, ytrain)
    five_best_features = list(xtrain.columns[sel_five_cols.get_support()])
    return five_best_features


def modelling_data_linReg_n_Feature_selection(xtrain, ytrain, k_feat, forward):
    """This function is used to find k best features using linear regresser
    for the given xtrain, ytrain using forward/backward algo.

    param xtrain: splitted train dataset without dependent column in the form
    of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    param k_feat: no. of features to be selected.

    type k_feat: int

    param forward: True for forward algo/False for backward algo

    type forward: boolean

    return: list of k best features

    rtype: list
    """

    log_reg = LogisticRegression()
    sfs_logReg = sfs(log_reg, k_features=10, forward=forward, verbose=2,
                     scoring='neg_mean_squared_error')
    sfs_logReg = sfs_logReg.fit(xtrain, ytrain)
    feat_names = list(sfs_logReg.k_feature_names_)
    return feat_names


def modelling_data_logReg_n_Feature_selection(xtrain, ytrain, k_feat, forward):
    """This function is used to find k best features using logistic regresser
    for the given xtrain, ytrain using forward/backward algo.

    param xtrain: splitted train dataset without dependent column in the form
    of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    param k_feat: no. of features to be selected.

    type k_feat: int

    param forward: True for forward algo/False for backward algo

    type forward: boolean

    return: list of k best features
    rtype: list
    """
    # Finding Best features using Backward Algorithm with Linear Regression
    lin_reg = LinearRegression()
    sfs_linReg = sfs(lin_reg, k_features=10, forward=forward, verbose=2,
                     scoring='neg_mean_squared_error')
    sfs_linReg = sfs_linReg.fit(xtrain, ytrain)
    feat_names = list(sfs_linReg.k_feature_names_)
    return feat_names


def splitdf_into_Xtrain_n_Ytrain(df, dep_col):
    """This function is used to split dataframe as independent and dependent
     dataframes.

    param df: dataset as dataframe

    type df: pandas dataframe

    param dep_col: dependent column name i.e output

    type dep_col: string

    return: xtrain i.e independent columns of the input dataframe

    rtype: pandas dataframe

    return: ytrain i.e dependent column of the input dataframe

    rtype: pandas dataframe
    """

    Ytrain = df[dep_col]
    Xtrain = df.drop([dep_col], axis=1)
    return Xtrain, Ytrain
