import numpy as np
from PIL import Image
from pydantic import BaseModel, ValidationError, validator
from fastapi import FastAPI, File, UploadFile, HTTPException, Depends
from steps.model_step.model import MNIST, get_model

route = FastAPI()


# class PredictRequest(BaseModel):
#     data: int
#
#     @validator("data")
#     def check_dimensionality(cls, v):
#         for point in v:
#             if len(point) != n_features:
#                 raise ValueError(f"Each data point must contain {n_features} features")
#
#         return v


class PredictResponse(BaseModel):
    prediction: int


@route.post("/predict",
            description="Predict MNIST image")
async def predict(file: UploadFile = File(...), model: MNIST = Depends(get_model)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format."
    await file.read()
    image = np.array(Image.open(file.file))

    y_pred = model.predict_single_image(image=image)
    result = PredictResponse(prediction=y_pred.tolist())
    print(y_pred)

    return result


