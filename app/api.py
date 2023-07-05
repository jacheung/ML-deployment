import numpy as np
from dotenv import load_dotenv
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from steps.model_step.model import MNIST

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

    # load model and predict
    mnist = MNIST(mlflow_registered_model_name='mnist-hyperparam-local')
    if mnist._model is None:
        return "No models found. Cannot parse request"
    y_pred = mnist.predict(context=None, model_input=image)
    result = PredictResponse(prediction=y_pred.tolist())
    print(y_pred)

    return result


