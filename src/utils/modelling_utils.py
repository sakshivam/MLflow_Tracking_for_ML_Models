import numpy as np
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
import lightgbm as lgbm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score


def test_train_split(df, frac_train):
    """This function is used to split dataframe as train and test dataframes.

    param df: dataset as dataframe

    type df: pandas dataframe

    param frac_train: fraction of which input dataframe is splitted as
     train dataframe

    type frac_train: float

    return: train_df i.e train dataframe from input dataframe

    rtype: pandas dataframe

    return: test_df i.e test dataframe from input dataframe

    rtype: pandas dataframe
    """

    frac = frac_train
    train_df = df.sample(frac=frac)
    test_df = df.drop(train_df.index)
    return train_df, test_df


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


def statsmodel_logit(xtrain, ytrain, xvalidation, yvalidation):
    """This function is used to build model using statsmodel for the given
     xtrain, ytrain

    param xtrain: train dataset without dependent column in the form
     of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    param xvalidation: validation dataset without dependent column in the form
     of dataframe

    type xvalidation: pandas dataframe

    param yvalidation: dependent column of validation dataset i.e output

    type yvalidation: pandas dataframe

    return: logit_model
    rtype: pickle
    """

    logit_model = sm.Logit(ytrain, xtrain).fit()
    print(logit_model.summary2())
    y_pred = logit_model.predict(xvalidation)
    prediction = list(map(round, y_pred))
    print(accuracy_score(yvalidation, prediction))
    return logit_model


def model_logit_sklearn(xtrain, ytrain, xvalidation, yvalidation):
    """This function is used to build model using sklearn logit model for the
     given xtrain, ytrain

    param xtrain: train dataset without dependent column in the form
    of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    param xvalidation: validation dataset without dependent column in the form
     of dataframe

    type xvalidation: pandas dataframe

    param yvalidation: dependent column of validation dataset i.e output

    type yvalidation: pandas dataframe

    return: logit_model

    rtype: pickle
    """

    logreg = LogisticRegression().fit(xtrain, ytrain)
    y_pred = logreg.predict(xvalidation)
    print(accuracy_score(yvalidation, y_pred))
    return logreg


def balancing_data_with_SMOTE(xtrain, ytrain):
    """This function is used to balance data using SMOTE for the given
     xtrain, ytrain

    param xtrain: train dataset without dependent column in the form
     of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    return: balanced xtrain

    rtype: pandas dataframe

    return: balanced ytrain

    rtype: pandas dataframe
    """

    # if the dataset is unbalanced i.e no. of values in each level of target
    # variable varies a lot.Then it should be balanced.
    # For this SMOTE is used.
    sm = SMOTE(random_state=42)
    X_train_sm, y_train_sm = sm.fit_resample(xtrain, ytrain)
    print(y_train_sm.value_counts())
    return X_train_sm, y_train_sm


def GridSearchCV_forBestParams(xtrain, ytrain):
    """This function is used to find best hyperparameters using Grid Search CV
     for the given xtrain, ytrain

    param xtrain: train dataset without dependent column in the form
    of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    return: best_params in the form dictionary

    rtype: dict
    """

    # Input Xtrain and Ytrain should be balanced
    gbm1 = lgbm.LGBMClassifier({
        'early_stopping_rounds': 100, 'bagging_fraction': 0.5,
        'bagging_freq': 5,  'objective': 'binary', 'num_boost_round': 1000,
        'num_threads': 4,
    })
    gridParams = {
        'boosting_type': ['dart', 'gbdt'],
        'metric': [
            'binary_logloss',
            'binary_error',
            'auc'
        ],
        'num_leaves': np.array([8, 16, 32]),
        'learning_rate': [0.05, 0.1, 0.15, 0.2, 0.25],
        'max_depth': [2, 3, 4, 5]
    }
    grid = GridSearchCV(gbm1, gridParams, verbose=2, cv=4, n_jobs=1)
    grid.fit(xtrain, ytrain)
    best_params = grid.best_params_
    return best_params


def lgbm_model(xtrain, ytrain, xvalidation, yvalidation):
    """This function is used to build model using lgbm model for the
    given xtrain, ytrain

    param xtrain: train dataset without dependent column in the form
     of dataframe

    type xtrain: pandas dataframe

    param ytrain: dependent column of dataframe i.e output

    type ytrain: pandas dataframe

    param xvalidation: validation dataset without dependent column in the form
     of dataframe

    type xvalidation: pandas dataframe

    param yvalidation: dependent column of validation dataset i.e output

    type yvalidation: pandas dataframe

    return: lgbm_model as gbm

    rtype: pickle
    """

    # Using LGBM Classifier with best parameters and then training the
    #  training data
    # Input Xtrain and Ytrain should be balanced
    grid_params = GridSearchCV_forBestParams(xtrain, ytrain)
    parameters = {'objective': 'binary', 'num_boost_round': 1000,
                  'num_threads': 4}
    parameters.update(grid_params)
#     param_str = str(parameters)
#     param_str =param_str.replace(':','=')
#     param_str = param_str.replace('{', '')
#     param_str = param_str.replace('}', '')
    gbm = lgbm.LGBMClassifier(**parameters)
    gbm.fit(xtrain, ytrain,
            early_stopping_rounds=100,
            eval_set=[(xtrain, ytrain), (xvalidation, yvalidation)])
    # y_pred = gbm.predict(xvalidation)
    # y_pred_prob = gbm.predict_proba(xvalidation)
    print('Accuracy for Train Set', cross_val_score(gbm, xtrain, ytrain,
                                                    scoring='accuracy',
                                                    cv=4, n_jobs=4))
    print('Accuracy for Validation Set', cross_val_score(gbm, xvalidation,
                                                         yvalidation,
                                                         scoring='accuracy',
                                                         cv=4, n_jobs=4))
    return gbm
