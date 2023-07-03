import tensorflow as tf
import tensorflow_hub as hub
import mlflow
import numpy as np
# Project imports
from steps.preprocess_step.preprocess import preprocess_mnist_tfds


class MNIST(mlflow.pyfunc.PythonModel): 
    def __init__(self, mlflow_registered_model_name: str = None):
        self._model = None
        self._mlflow_registered_model_name = mlflow_registered_model_name
        self.load()    
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

    def fit_hp_search(self, xy_train, xy_test, hyperparameters):                      
        self._build(self, hyperparameters)
        # fit model using train/test split to find hyperparams
        self._train_history = self._model.fit(xy_train,
                                               epochs=hyperparameters['epochs'],
                                               validation_data=xy_test)
    
    def fit_production(self, xy_train, hyperparameters):                      
        self._build(self, hyperparameters)
        # fit model using all the data 
        self._train_history = self._model.fit(xy_train,
                                               epochs=hyperparameters['epochs'])
        
    def load(self):
        try:
            results = mlflow.search_registered_models(
                filter_string=f'name = "{self._mlflow_registered_model_name}"')
            latest_model_details = results[0].latest_versions[0]
            self._model = mlflow.pyfunc.load_model(
                model_uri=f'{latest_model_details.source}')
        except FileNotFoundError:
            print('No models found.')
            self._model = None
            return self
        
    def predict(self, context, model_input: np.ndarray) -> np.ndarray:
        image, _ = preprocess_mnist_tfds(model_input)
        image = tf.reshape(image, [1, 224, 224, 3])
        return self._model.predict(image).argmax()


def get_model():
    model = MNIST()
    return model