import pickle
import pandas as pd
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve(strict=True).parent.parent

with open(f"{BASE_DIR}/config.json") as f:
    config = json.load(f)

__version__ = config["model_version"]

with open(f"{BASE_DIR}/model/model.pkl", "rb") as f:
    model = pickle.load(f)

demographics = pd.read_csv(f"{BASE_DIR}/model/zipcode_demographics.csv",
                                   dtype={'zipcode': str})
feature_list = pd.read_json(f"{BASE_DIR}/model/model_features.json").values.ravel().tolist()

def model_predict(input_data):
    input_data["zipcode"] = input_data["zipcode"].astype(str)
    merged_data = input_data.merge(demographics, how="left",
                             on="zipcode").drop(columns="zipcode")
    model_features = merged_data.reindex(columns=feature_list)
    return model.predict(model_features)