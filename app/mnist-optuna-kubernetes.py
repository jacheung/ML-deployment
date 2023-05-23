import os
import time
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_datasets as tfds
import mlflow
import optuna
from optuna.integration.mlflow import MLflowCallback



def load_tensorflow_dataset(dataset_str: str):
    (xy_train, xy_test), ds_info = tfds.load(
        dataset_str,
        split=['train', 'test'], shuffle_files=True,
        as_supervised=True,
        with_info=True,
    )
    return (xy_train, xy_test)


def preprocess_mnist_tfds(image, label=None):
    # reshape and upsample to 3 channel for transfer learning models
    # ... for when no channel information is present
    if len(image.shape) != 3:
        image = np.dstack((image, image, image))
    # ... for when channel is only 1 dimension
    if image.shape[2] == 1:
        image = tf.image.grayscale_to_rgb(image)
    # normalize pixel values
    image = tf.cast(image, tf.float32) / 255.
    # resize with pad for mobilenetv2
    image = tf.image.resize_with_pad(image, target_height=224, target_width=224)
    return image, label


class MNIST(mlflow.pyfunc.PythonModel):     
    def fit(self, xy_tuple_train, xy_tuple_test, hyperparameters):
        ## Build model
        # class names for mnist hardcoded
        class_names = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    
        # set layer regularization for DNN
        regularizer = tf.keras.regularizers.l1_l2(hyperparameters['l1'], hyperparameters['l2'])

        # load in mobilenetv2 weights and instantiate dense classification head 
        base_model = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
        layers = [
            hub.KerasLayer(
                base_model,
                input_shape=(224, 224, 3),
                trainable=False,
                name='mobilenet_embedding'),
            tf.keras.layers.Dense(hyperparameters['num_hidden'],
                                  kernel_regularizer=regularizer,
                                  activation='relu',
                                  name='dense_hidden'),
            tf.keras.layers.Dense(len(class_names),
                                  kernel_regularizer=regularizer,
                                  activation='softmax',
                                  name='mnist_prob')
        ]

        self._model = tf.keras.Sequential(layers, name='mnist-classification')

        # compile model 
        self._model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hyperparameters['learning_rate']),
                            loss=tf.keras.losses.SparseCategoricalCrossentropy(
                            from_logits=False),
                            metrics=['accuracy'])
                      
        ## Fit model
        # fit model and save history to model store
        self._train_history = self._model.fit(xy_tuple_train, epochs=hyperparameters['epochs'], validation_data=xy_tuple_test)
        self._model_base = base_model
        
    def predict(self, context, model_input: np.ndarray) -> np.ndarray:
        image, _ = preprocess_mnist_tfds(model_input)
        image = tf.reshape(image, [1, 224, 224, 3])
        return self._model.predict(image).argmax()


# mlflow Tracking requires definition of experiment name AND logged params
# Experiment names they should be defined as "project-task-version"
def set_mlflow_experiment(experiment_name:str):
    try:
        experiment_id = mlflow.create_experiment(experiment_name)
    except mlflow.exceptions.MlflowException as e: 
        if str(e) == f"Experiment '{experiment_name}' already exists.":
            print(f'Experiment already exists, setting experiment to {experiment_name}')
            experiment_info = mlflow.set_experiment(experiment_name)
            experiment_id = experiment_info.experiment_id
    experiment = mlflow.get_experiment(experiment_id)
    print("---------------------")
    print('Experiment details are:')
    print("Name: {}".format(experiment.name))
    print("Experiment_id: {}".format(experiment.experiment_id))
    print("Artifact Location: {}".format(experiment.artifact_location))
    print("Creation timestamp: {}".format(experiment.creation_time))
    return experiment_id



# hyperparameters search using Optuna
# can scale Optuna with Kubernetes https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/004_distributed.html
def objective(trial): 
    """
    Optuna objective function for tuning transfer learning model
    """
    hyperparams = {
        'learning_rate': trial.suggest_float('learning_rate', 0.00001, 0.1, log=True),
        'l1': trial.suggest_float('l1', 0.0, 0.05),
        'l2': trial.suggest_float('l2', 0.0, 0.05),
        'num_hidden': trial.suggest_int('num_hidden', 8, 64),
        'epochs': trial.suggest_int('epochs', 1, 3)
    }

    model.fit(ds_train, ds_test, hyperparams)
    training_history = model._train_history.history
    validation_accuracy = training_history['val_accuracy'][-1]
    return validation_accuracy




if __name__ == "__main__":
    # preprocess and define batch sizes for tensorflow 
    ds_train, ds_test = load_tensorflow_dataset('mnist')
    ds_train = ds_train.map(preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(128)
    ds_test = ds_test.map(preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_test = ds_test.batch(128) 

    # instantiate model
    model = MNIST()

    # define optuna variables
    optuna_study_name = "mnist-hyperparam-optuna-k8s"
    optuna_storage_url="postgresql://{}:{}@localhost:5432/{}".format(
                os.environ["POSTGRES_USER"],
                os.environ["POSTGRES_PASSWORD"],
                os.environ["POSTGRES_DB"],
            )

    # create or load optuna study
    try:
        print('loading study...')
        study = optuna.load_study(
            study_name=optuna_study_name,
            storage=optuna_storage_url,
        )
        
    except KeyError:
        print('no study found. building from scratch...')
        study = optuna.create_study(
            study_name=optuna_study_name,
            storage=optuna_storage_url,
            pruner=optuna.pruners.HyperbandPruner(),
            direction='maximize')


    # create or set an experiment for optuna. Each trial from Optuna is logged as one run in an MLFlow experiment.
    experiment_id = set_mlflow_experiment(experiment_name=optuna_study_name)
    mlflow_kwargs = {'experiment_id': experiment_id}

    # a new experiment name will be created in MLFlow using the Optuna study name
    study.optimize(objective,
                n_trials=8,
                n_jobs=2,
                callbacks=[MLflowCallback(metric_name="val_accuracy",
                                            create_experiment=False,
                                            mlflow_kwargs=mlflow_kwargs)]
                )