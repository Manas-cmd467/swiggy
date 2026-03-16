from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.pipeline import Pipeline
import uvicorn
import pandas as pd
import mlflow
import json
import joblib
from mlflow import MlflowClient
from sklearn import set_config
from scripts.data_clean_utils import perform_data_cleaning

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn import set_config

# cleaning function
from scripts.data_clean_utils import perform_data_cleaning

# sklearn should return pandas
set_config(transform_output="pandas")

# =========================
# Load model and preprocessor
# =========================

model = joblib.load("models/model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")

model_pipe = Pipeline([
    ("preprocess", preprocessor),
    ("regressor", model)
])

# =========================
# FastAPI app
# =========================

app = FastAPI()

# =========================
# Input schema
# =========================

class Data(BaseModel):
    ID: str
    Delivery_person_ID: str
    Delivery_person_Age: str
    Delivery_person_Ratings: str
    Restaurant_latitude: float
    Restaurant_longitude: float
    Delivery_location_latitude: float
    Delivery_location_longitude: float
    Order_Date: str
    Time_Orderd: str
    Time_Order_picked: str
    Weatherconditions: str
    Road_traffic_density: str
    Vehicle_condition: int
    Type_of_order: str
    Type_of_vehicle: str
    multiple_deliveries: str
    Festival: str
    City: str

# =========================
# Routes
# =========================

from fastapi.responses import RedirectResponse

@app.get("/")
def home():
    return RedirectResponse("/docs")

@app.post("/predict")
def predict(data: Data):

    df = pd.DataFrame({
        "ID": data.ID,
        "Delivery_person_ID": data.Delivery_person_ID,
        "Delivery_person_Age": data.Delivery_person_Age,
        "Delivery_person_Ratings": data.Delivery_person_Ratings,
        "Restaurant_latitude": data.Restaurant_latitude,
        "Restaurant_longitude": data.Restaurant_longitude,
        "Delivery_location_latitude": data.Delivery_location_latitude,
        "Delivery_location_longitude": data.Delivery_location_longitude,
        "Order_Date": data.Order_Date,
        "Time_Orderd": data.Time_Orderd,
        "Time_Order_picked": data.Time_Order_picked,
        "Weatherconditions": data.Weatherconditions,
        "Road_traffic_density": data.Road_traffic_density,
        "Vehicle_condition": data.Vehicle_condition,
        "Type_of_order": data.Type_of_order,
        "Type_of_vehicle": data.Type_of_vehicle,
        "multiple_deliveries": data.multiple_deliveries,
        "Festival": data.Festival,
        "City": data.City
    }, index=[0])

    cleaned = perform_data_cleaning(df)
    prediction = model_pipe.predict(cleaned)[0]

    return {"predicted_delivery_time_minutes": float(prediction)}

# =========================
# Run server
# =========================

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)