from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import pandas as pd
from datetime import datetime
from app.model.model import model_predict
from app.model.model import __version__ as model_version

app = FastAPI()

class InputData(BaseModel):
    ID: Optional[int] = None
    date: Optional[str] = None
    bedrooms: int
    bathrooms: float
    sqft_living: int
    sqft_lot: int
    floors: float
    waterfront: int
    view: int
    condition: int	
    grade: int
    sqft_above: int
    sqft_basement: int
    yr_built: int
    yr_renovated: int
    zipcode: int
    latitude: float
    longitude: float
    sqft_living15: int
    sqft_lot15: int

class PredictionOut(BaseModel):
    home_value: float
    pred_time: str
    model_version: str

# TODO: path should include model version
@app.post("/predict/{model_version}", response_model=PredictionOut)
def predict(payload: InputData):
    input_df = pd.DataFrame([payload.dict()])
    home_value = model_predict(input_df)
    cur_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    prediction = {
        "home_value": home_value,
        "pred_time": cur_time,
        "model_version": model_version
        }
    return prediction