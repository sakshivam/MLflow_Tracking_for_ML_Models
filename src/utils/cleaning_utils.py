def Columns_to_drop(df):
    """This function is used to find the column names having percent
    of missing data greater tha 95 %.

    param df: dataset whose columns with missing data > 95% to be dropped.

    type df: pandas dataframe

    return: list of columns

    rtype: list
    """
    # null_count = df.isna().sum()
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_val_columns = list(percent_missing[percent_missing > 95].index)
    return missing_val_columns


def Missing_Val_Columns_to_fill(df):
    """This function is used to find the column names having missing data
    percentage is between 0 and 95%.

    param df: dataset whose columns with missing data between 0 and 95% to
    be filled with some values.

    type df: pandas dataframe

    return: list of columns

    rtype: list
    """

    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_val_col_list_to_fill = list(percent_missing[(percent_missing > 0)
                                        & (percent_missing < 95)].index)

    return missing_val_col_list_to_fill


def Drop_Missing_Val_Columns(df, col_list_to_drop):
    """This function is used to drop the column names which are provided as
    parameters.

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
    """This function is used to fill the column names which are provided as
     parameters
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
