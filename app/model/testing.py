import pandas as pd
import json
from pathlib import Path
import pdb

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

feature_list = pd.read_json(f"{BASE_DIR}/model/model_features.json")
pdb.set_trace()

"""
curl -X 'POST' \
  'http://0.0.0.0/predict/0.1.0' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "bedrooms": 0,
  "bathrooms": 0,
  "sqft_living": 0,
  "sqft_lot": 0,
  "floors": 0,
  "waterfront": 0,
  "view": 0,
  "condition": 0,
  "grade": 0,
  "sqft_above": 0,
  "sqft_basement": 0,
  "yr_built": 0,
  "yr_renovated": 0,
  "zipcode": 98001,
  "latitude": 0,
  "longitude": 0,
  "sqft_living15": 0,
  "sqft_lot15": 0
}'

curl -X 'POST' \
  'http://0.0.0.0/predict/0.1.1' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "bedrooms": 0,
  "bathrooms": 0,
  "sqft_living": 0,
  "sqft_lot": 0,
  "floors": 0,
  "waterfront": 0,
  "view": 0,
  "condition": 0,
  "grade": 0,
  "sqft_above": 0,
  "sqft_basement": 0,
  "yr_built": 0,
  "yr_renovated": 0,
  "zipcode": 98001,
  "latitude": 0,
  "longitude": 0,
  "sqft_living15": 0,
  "sqft_lot15": 0
}'

"""