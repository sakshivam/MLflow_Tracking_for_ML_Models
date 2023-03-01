# MLFlow tracking
This is an API along with UI which can be used to log parameters, metrics, code versions, and output files so that we can we can visualize results of our ML codes later on.

To perform MLflow tracking, i.e we should understand the concept of runs. Each time our code gets executed it will be considered as a run. With each run, following information gets recorded:

1. Code Version : Git commit hash when run from an MLflow Project.

2. Start & End Time of the run

3. Source : Name of the file to launch the run, or the project name and entry point for the run from an MLflow Project.

4. Parameters : Key-value input parameters in the format of strings.

5. Metrics : Key-value metrics( Here value is numerical)

6. Artifacts : Output files in any format. we can record images, models and data files as artifacts.

MLflow runs can be recorded to local files, to a SQLAlchemy compatible database, or remotely to a tracking server. By default, the MLflow Python API logs runs locally to files in an mlruns directory in the same directory where project gets executed. You can then run mlflow ui to see the logged runs.

To log runs remotely, set the MLFLOW_TRACKING_URI environment variable to a tracking server’s URI or call mlflow.set_tracking_uri().

Steps to perform MLFlow Tracking on local:
1. we first need to set MLFLOW_TRACKING_URI to either local file path.
2. Then we need to create experiment and get its id by first checking whether it has already been created or not. If already created use mlflow.get_experiment_by_name otherwise use mlflow.create_experiment
3. Start a run using the following command:

with mlflow.start_run(experiment_id=experiment_id, run_name ="any name") as run:

4. Now we can log models using mlflow.log_model if model is sklearn based use mlflow.sklearn.log_model. For model we should use get_tracking_uri to use this model further.
Also we can log metrics using mlflow.log_metric.





