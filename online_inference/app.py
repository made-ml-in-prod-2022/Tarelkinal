import logging
import os

from dotenv import find_dotenv, load_dotenv
import joblib
from typing import List, Union, Optional

import pandas as pd
import uvicorn
from fastapi import FastAPI, status, Response
from pydantic import BaseModel, conlist, validator
from sklearn.pipeline import Pipeline


logger = logging.getLogger(__name__)


def load_object(path: str) -> Pipeline:
    model = joblib.load(path)
    return model


class HeartDiseaseModel(BaseModel):
    data: List[conlist(Union[float, int, None], min_items=13, max_items=13)]
    features: List[str]


class DiseaseProbResponse(BaseModel):
    data: List[conlist(Union[float, int], min_items=13, max_items=13)]
    features: List[str]
    prob: float

    @validator('prob')
    def prob_check(cls, v):
        assert 0 <= v <= 1
        return v


model: Optional[Pipeline] = None


def make_predict(
        data: List, features: List[str], model: Pipeline,
) -> DiseaseProbResponse:
    df = pd.DataFrame(data, columns=features)
    prob = model.predict_proba(df)[:, 1][0]

    result = DiseaseProbResponse(
        data=data,
        features=features,
        prob=prob
    )
    return result


app = FastAPI()


@app.get("/")
def main():
    return "it is entry point of our predictor"


@app.on_event("startup")
def load_model():
    global model
    model_path = os.getenv("PATH_TO_MODEL")
    if model_path is None:
        err = f"PATH_TO_MODEL {model_path} is None"
        logger.error(err)
        raise RuntimeError(err)

    model = load_object(model_path)


@app.get("/healz")
def health(response: Response):
    if model is None:
        response.status_code = status.HTTP_400_BAD_REQUEST
    else:
        response.status_code = status.HTTP_200_OK

    return response.status_code


@app.get("/predict/", response_model=DiseaseProbResponse)
def predict(request: HeartDiseaseModel):

    return make_predict(request.data, request.features, model)


if __name__ == "__main__":
    dotenv_path = find_dotenv()
    load_dotenv(dotenv_path)
    uvicorn.run("app:app", host="0.0.0.0", port=os.getenv("PORT", 8000))
