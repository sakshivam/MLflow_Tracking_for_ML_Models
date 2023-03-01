import pandas as pd
from joblib import load


def Drop_Missing_Val_Columns(df, col_list_to_drop):
    """This function is used to drop the column names which are
    provided as parameters.

    param df: dataset in which required columns are to be dropped.

    type df: pandas dataframe

    param col_list_to_drop: list of columns to drop

    type col_list_to_drop : list

    return: new dataset after removing required columns.

    rtype: pandas dataframe
    """

    df.drop(columns=col_list_to_drop, inplace=True)
    return df


def Fill_missing_Val_Columns(df, col_list_to_fill):
    """This function is used to fill the column names which are provided
     as parameters
    with most frequent value in that column.

    param df: dataset in which required columns are to be filled.

    type df: pandas dataframe

    param col_list_to_fill: list of columns to fill

    type col_list_to_fill : list

    return: new dataset after removing required columns.

    rtype: pandas dataframe
    """

    for i in col_list_to_fill:
        df[i] = df[i].fillna(df[i].value_counts().index[0])
        # print(df[i].isna().sum())
    return df


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


def split_into_XnY(df, dep_col):
    """This function is used to split dataframe as independent and
    dependent dataframes.

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


def predictions(testdf, dep_col, features, model_file_path):
    """This function is used to predict the output or dependent feature using model

    param testdf: test dataset in the form of dataframe.

    type testdf: pandas dataframe

    param dep_col: dependent column name i.e output

    type dep_col: string

    param features: list of best features

    type features : list

    param model_file_path: path where model is saved as pickle

    type model_file_path: path string

    return: resultant dataset

    rtype: pandas dataframe
    """

    Xtest, Ytest = split_into_XnY(testdf, dep_col)
    Xtest_tenfeat = Xtest[features]
    mdl = load(model_file_path)
    y_pred = mdl.predict(Xtest_tenfeat)
    y_pred_df = pd.DataFrame(y_pred)
    y_pred_df.columns = ['Y_Predicted']
    merged_df = pd.concat([Xtest, y_pred_df], axis=1)
    return merged_df
