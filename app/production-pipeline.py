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
        shuffle_files=True,
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
    @staticmethod
    def _build(self, hyperparameters):
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
        
        # base model logging
        self._model_base = base_model

    def fit_hp_tuning(self, xy_tuple_train, xy_tuple_test, hyperparameters):                      
        self._build(hyperparameters)
        # fit model using train/test split to find hyperparams
        self._train_history = self._model.fit(xy_tuple_train, epochs=hyperparameters['epochs'], validation_data=xy_tuple_test)
    
    def fit_production(self, xy_tuple_train, hyperparameters):                      
        self._build(hyperparameters)
        # fit model using all the data 
        self._train_history = self._model.fit(xy_tuple_train, epochs=hyperparameters['epochs'])
        
    def predict(self, context, model_input: np.ndarray) -> np.ndarray:
        image, _ = preprocess_mnist_tfds(model_input)
        image = tf.reshape(image, [1, 224, 224, 3])
        return self._model.predict(image).argmax()
    
    def load_from_mlflow(self, model_name):
        results = mlflow.search_registered_models(filter_string='name = "mnist-classification"')
        latest_model_details = results[0].latest_versions[0]
        model = mlflow.pyfunc.load_model(model_uri=f'{latest_model_details.source[7:]}')


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


if __name__ == "__main__":
    # args to parse
    train = True
    mlflow_experiment_name = "mnist-hyperparam-optuna-docker"
    
    # preprocess and define batch sizes for tensorflow 
    ds_train = load_tensorflow_dataset('mnist')
    ds_train = ds_train.map(preprocess_mnist_tfds, num_parallel_calls=tf.data.AUTOTUNE)
    ds_train = ds_train.batch(128)

    # instantiate model
    model = MNIST()

    # create or set an experiment for optuna. Each trial from Optuna is logged as one run in an MLFlow experiment.
    experiment_id = set_mlflow_experiment(experiment_name=mlflow_experiment_name)
    
    # train model using params or load if model is registered? CREATE ARG FOR THIS
    if train is True:
        # load params 
        hyperparams = 1 # load from mlflow experiment
        model.fit(ds_train, ds_test, hyperparams)
        # register model
        model_name = f'{experiment.name}'
        # mlflow.tensorflow.log_model
        mv = mlflow.register_model(model_uri=f"runs:/{best_run_id}/",
                                   name=model_name)
    else:
        model.load_mlflow()


    # infer model


    