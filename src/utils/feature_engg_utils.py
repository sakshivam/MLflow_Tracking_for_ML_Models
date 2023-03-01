import numpy as np


def Categorical_Numerical_Features_split(df):
    """This function is used to split the columns as categorical
    and numerical columns.

    param df: dataset as dataframe

    type df: pandas dataframe

    return: list of columns with categorical columns

    rtype: list

    return: list of columns with numerical columns

    rtype: list
    """
    categorical_data = df.select_dtypes(exclude=[np.number])
    cat_list = list(categorical_data.columns)
    numeric_data = df.select_dtypes(include=[np.number])
    num_list = list(numeric_data.columns)
    return cat_list, num_list


def plot_stackedgraph_categorical(df, cat_list):
    """This function is used to plot list of features as stacked bar graph
    for all classes inside a particular column.

    param df: dataset as dataframe

    type df: pandas dataframe

    param cat_list: list of columns to be plotted

    type cat_list: list
    """
    for i in cat_list:
        df.groupby([i])['Y'].apply(lambda x: x.value_counts() / len(x))\
          .transpose().unstack().plot(kind='bar', stacked=True)


def scatterplot_for_numerical_features(df, num_lst):
    """This function is used to plot list of features as scatterplot
    for all columns which are provided in the list.

    param df: dataset as dataframe

    type df: pandas dataframe

    param num_lst: list of columns to be plotted

    type num_lst: list
    """
    num_list_new = num_lst[0:7]
    for i in num_list_new:
        df.plot.scatter(x=i, y='Y', c='DarkBlue')


def replace_values_using_dict(df, dict_to_replace_values):
    """This function is used to replace the column values with values provided
     in the dictionary as key value pairs.

    param df: dataset in which required columns are to be filled.

    type df: pandas dataframe

    param dict_to_replace_values: dictionary with key value pairs- which value
    is to be replaced by whom.

    type dict_to_replace_values : dict

    return: new dataset after replacing values.

    rtype: pandas dataframe
    """
    df = df.replace(dict_to_replace_values)
    return df
