import tensorflow as tf
import optuna
# Project Imports
from steps.load_step import load
from steps.preprocess_step import preprocess
from steps.model_step import model
from steps import utils



if __name__ == "__main__":
    # args to parse
    train = True
    mlflow_experiment_name = "mnist-hyperparam-optuna-docker"


    # preprocess and define batch sizes for tensorflow 
    ds_train = load.load_tensorflow_dataset_production('mnist')
    ds_train = ds_train.map(preprocess.preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(128)

    # instantiate model
    mnist_model = model.MNIST()

    # train model using params or load if model is registered? CREATE ARG FOR THIS
    if train is True:
        # load params 
        optuna_study_name = "mnist-hyperparam-optuna-docker"
        # optuna_storage_url="postgresql://{}:{}@{}:5432/{}".format(
        #             os.environ["POSTGRES_USER"],
        #             os.environ["POSTGRES_PASSWORD"],
        #             os.environ["POSTGRES_OPTUNA_HOSTNAME"],
        #             os.environ["POSTGRES_OPTUNA_DB"]
        #         )
        POSTGRES_DB="optunadb"
        POSTGRES_USER="postgres"
        POSTGRES_PASSWORD="mysecretpassword"
        POSTGRES_HOSTNAME="localhost"
        optuna_storage_url="postgresql://{}:{}@{}:5432/{}".format(
                POSTGRES_USER,
                POSTGRES_PASSWORD,
                POSTGRES_HOSTNAME,
                POSTGRES_DB
                )
        print('loading study...')
        study = optuna.load_study(
            study_name=optuna_study_name,
            storage=optuna_storage_url,
        )  
        hyperparameters = study.best_params
        mnist_model.fit_production(xy_train=ds_train,
                                   hyperparameters=hyperparameters)
    else:
        mnist_model.load()


    # infer model


    