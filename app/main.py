from fastapi import FastAPI
from starlette.responses import JSONResponse
from joblib import load
import pandas as pd

app = FastAPI()

sgd_pipe = load('../models/sgd_pipeline.joblib')

@app.get("/")
def read_root():
    return {"Description": "This model predicts the sales revenue for a given item in a specific store of a given date. There are total 10 stores across 3 states: Califormnia(CA),Texas(TX) and Wisconsin(WI). Each shop sells items from 3 different categories: hobbies, foods and household.",
            "list_of_endpoints": "'/': Displays a brief description of the project, '/health': Welcome message,'/sales/stores/items/': return predicted sales volumne for an input item, store an date",
            "data_format":"store_id: should be in format like WI_#, CA_#, TX#, item_id: should be in format categoryname_#_#, date: in a format yyyy-mm-dd",
            "github_repo_link":"https://github.com/sagarbhagwatkar/AT2"
            }

@app.get('/health', status_code=200)
def healthcheck():
    return 'So far so good'



def format_features(
    item_id: str,
    store_id: str,
    date: str,
    ):
    return {
        'item_id': [item_id],
        'store_id': [store_id],
        'date': [date]
    }



@app.get("/sales/stores/items/")
def predict(
    item_id: str,
    store_id: str,
    date: str,
):
    features = format_features(
        item_id,
        store_id,
        date,
        )
    obs = pd.DataFrame(features)
    pred = sgd_pipe.predict(obs)
    return JSONResponse(pred.tolist())











