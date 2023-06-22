import tensorflow as tf
import optuna
import os
import mlflow
import pandas as pd
# Project Imports
from steps.load_step import load
from steps.preprocess_step import preprocess
from steps.model_step import model
from steps import utils



if __name__ == "__main__":
    # arg parser for local
    train=True
    os.environ['MLFLOW_TRACKING_URI'] = "postgresql+psycopg2://postgres:mysecretpassword@localhost:5435/mlflowdb"
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9001"
    os.environ['AWS_ACCESS_KEY_ID'] = 'minio_user'
    os.environ['AWS_SECRET_ACCESS_KEY'] = "minio_pass"
    os.environ["POSTGRES_USER"] = 'postgres'
    os.environ["POSTGRES_PASSWORD"] = 'mysecretpassword'
    os.environ["POSTGRES_OPTUNA_HOSTNAME"] = 'localhost'
    os.environ["POSTGRES_OPTUNA_DB"] = 'optunadb'
    optuna_study_name = "mnist-hyperparam-local"


    # preprocess and define batch sizes for tensorflow 
    ds_train = load.load_tensorflow_dataset_production('mnist')
    ds_train = ds_train.map(preprocess.preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(128)

    # instantiate model
    mnist_model = model.MNIST()

    # mlflow experiment
    experiment_id = utils.set_mlflow_experiment(experiment_name=optuna_study_name,
                                                artifact_location='s3://mlflow')
    
    # train model using params or load if model is registered? CREATE ARG FOR THIS
    if train is True:
        # load params from optuna
        optuna_storage_url="postgresql://{}:{}@{}:5435/{}".format(
                    os.environ["POSTGRES_USER"],
                    os.environ["POSTGRES_PASSWORD"],
                    os.environ["POSTGRES_OPTUNA_HOSTNAME"],
                    os.environ["POSTGRES_OPTUNA_DB"]
                )
        print('loading study...')
        study = optuna.load_study(
            study_name=optuna_study_name,
            storage=optuna_storage_url,
        )  
        hyperparameters = study.best_params

        # train model and log via mlflow
        current = pd.to_datetime('now')
        mlflow_run_name=f'{optuna_study_name}-{current.year}{current.month}{current.day}'
        with mlflow.start_run(experiment_id=experiment_id,  
                              run_name=mlflow_run_name) as run:
            mnist_model.fit_production(xy_train=ds_train,
                                       hyperparameters=hyperparameters)
            # MLFlow Tracking parameters
            mlflow.log_params(params=hyperparameters)
    
            # MLFlow Tracking metrics 
            # Logging metrics for each epoch (housed in dictionary)
            training_history = mnist_model._train_history.history
            for epoch in range(0, hyperparameters['epochs']):
                insert = {}
                for metric, value in training_history.items():
                    insert[metric] = training_history[metric][epoch]
                mlflow.log_metrics(metrics=insert, step=epoch+1)

            # MLFlow tracking artifact (e.g. model file)
            # this will log the model and all its details under run_id/artifacts
            mlflow.pyfunc.log_model(python_model=mnist_model,
                                    artifact_path="")

            # Close out MLFlow run to prevent any log contamination.
            mlflow.end_run(status='FINISHED') 

    # infer model


    