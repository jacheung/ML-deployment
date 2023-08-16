import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import mlflow
# project imports
from steps.preprocess_step import preprocess


load_dotenv()
route = FastAPI()

class PredictResponse(BaseModel):
    prediction: int


@route.post("/predict",
            description="Predict MNIST image")
async def predict(file: UploadFile = File(...)):
    # file check and load
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format."
    await file.read()
    image = np.array(Image.open(file.file))

    # load model from mlflow
    try: 
        results = mlflow.search_registered_models(
            filter_string='name = "mnist-hyperparam-local"')
        latest_model_details = results[0].latest_versions[0]
        model = mlflow.tensorflow.load_model(
            model_uri=f'{latest_model_details.source}')
        print(f'Successfully loaded model from {latest_model_details.source}')
    except IndexError:
        print('No models found. Cannot perform inference.')
        return None

    # preprocess image and predict
    image, _ = preprocess.preprocess_mnist_tfds(image)
    image = tf.reshape(image, [1, 224, 224, 3])
    result = model.predict(image).argmax()

    # process output 
    result = PredictResponse(prediction=result.tolist())
    print(result)

    return result


