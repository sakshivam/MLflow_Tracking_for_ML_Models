import pandas as pd
from joblib import dump
import yaml
from src.utils.modelling_utils import (test_train_split,
                                       split_into_XnY,
                                       model_logit_sklearn)
from ml_service.utils.env_variables import Env
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import mlflow
import numpy as np


e = Env()

with open('.\\src\\utils\\config.yml') as file:
    try:
        config = yaml.safe_load(file)
        print('config read | Completed')
    except yaml.YAMLError:
        print('config read | Error')

# mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_tracking_uri("./mlruns")

traindf_with_feature_engg = pd.read_parquet(e.train_fengg_file_path)
print(traindf_with_feature_engg.head())

traindf_with_feature_engg.drop(config['DROP_COL_LIST_WITH_ONE_CLASS'],
                               axis=1, inplace=True)

traindf, validationdf = test_train_split(traindf_with_feature_engg,
                                         config['FRACTION_TRAIN'])
Xtrain, Ytrain = split_into_XnY(traindf, 'Y')
Xval, Yval = split_into_XnY(validationdf, 'Y')
Xtrain_tenfeat = Xtrain[config['TEN_BEST_FEATURES_OBSERVED_SELECTION']]
Xval_tenfeat = Xval[config['TEN_BEST_FEATURES_OBSERVED_SELECTION']]



model = model_logit_sklearn(Xtrain_tenfeat, Ytrain, Xval_tenfeat, Yval)
dump(model, e.model_file_path)
try:
    exp = mlflow.get_experiment_by_name('Logit_sklearn')
    experiment_id = exp.experiment_id
    print('Experiment Id is :', exp.experiment_id)
except:
    experiment_id = mlflow.create_experiment("Logit_sklearn")
    print('Experiment Id is :', experiment_id)

with mlflow.start_run(experiment_id=experiment_id, run_name ="predictions_metrics") as run:
    # model = model_logit_sklearn(Xtrain_tenfeat, Ytrain, Xval_tenfeat, Yval)
    # run_id = run.info.run_id
    y_pred = model.predict(Xval_tenfeat)

    accuracy = accuracy_score(Yval, y_pred)
    rmse = np.sqrt(mean_squared_error(Yval, y_pred))
    mae = mean_absolute_error(Yval, y_pred)
    r2 = r2_score(Yval, y_pred)
    print ("Accuracy is", accuracy)
    print("RMSE is", rmse)
    print("MAE is", mae)
    print("r2 score is",r2)

    mlflow.sklearn.log_model(model, "model_sklearn")
    model_uri = mlflow.get_artifact_uri("model_sklearn")
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("RMSE", rmse)
    mlflow.log_metric("MAE", mae)
    mlflow.log_metric("r2 score", r2)
