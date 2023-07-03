import numpy as np
import os
from PIL import Image
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from steps.model_step.model import MNIST

route = FastAPI()
# mlflow tracking server
os.environ['MLFLOW_TRACKING_URI'] = "http://0.0.0.0:5000"
# mlflow artifact/model store
os.environ['MLFLOW_S3_ENDPOINT_URL'] = "http://localhost:9000"
os.environ['AWS_ACCESS_KEY_ID'] = 'minio_user'
os.environ['AWS_SECRET_ACCESS_KEY'] = "minio_pass"


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
    model = MNIST(mlflow_registered_model_name='mnist-hyperparam-local')
    y_pred = model.predict(context=None, model_input=image)
    result = PredictResponse(prediction=y_pred.tolist())
    print(y_pred)

    return result


