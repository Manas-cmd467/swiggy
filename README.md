# Swiggy Delivery Time Prediction 🛵

An end-to-end MLOps project that predicts food delivery time using machine learning.  
The project demonstrates a production-style ML pipeline with data preprocessing, model training, experiment tracking, and deployment using FastAPI.

The trained model is exposed through a REST API that returns the predicted delivery time in minutes for a given order.

## 📌 Problem Statement

Food delivery platforms must accurately estimate delivery time to improve:
* Customer satisfaction
* Delivery partner efficiency
* Order management

This project builds a machine learning model that predicts delivery time based on order details, location, traffic, and delivery conditions.

## 📊 Features Used

Example features used by the model include:
* Delivery partner age & rating
* Restaurant latitude & longitude
* Delivery latitude & longitude
* Order date, time, and pickup time
* Weather conditions
* Road traffic density
* Vehicle condition & type
* Type of order
* Multiple deliveries
* Festival occurrences
* City type

## ⚙️ Machine Learning Pipeline

The ML pipeline is orchestrated using **DVC**.

**Pipeline stages:**
1. Data Cleaning
2. Data Preparation
3. Feature Engineering
4. Model Training
5. Model Evaluation

Run the full pipeline using:
```bash
dvc repro
```

## 🧠 Model Architecture

The final model is an ensemble **Stacking Regressor**.

* **Base Models:** Random Forest Regressor, LightGBM Regressor
* **Meta Model:** Linear Regression
* **Target Transformation:** PowerTransformer

**Pipeline Flow:**
> Raw Data ➡️ Data Cleaning ➡️ Feature Engineering ➡️ Random Forest + LightGBM ➡️ Stacking Regressor ➡️ Power Transformer (inverse) ➡️ **Predicted Delivery Time**

## 📦 Model Artifacts

The trained models are saved inside the `models/` directory and are loaded during API inference:
```text
models/
 ├── model.joblib
 ├── stacking_regressor.joblib
 ├── power_transformer.joblib
 └── preprocessor.joblib
```

## 📂 Project Structure

```text
swiggy-delivery-time-prediction/
├── app.py  
├── params.yaml  
├── dvc.yaml  
├── dvc.lock  
├── Dockerfile  
├── requirements-dev.txt  
├── requirements-dockers.txt  
├── data/  
│    ├── raw/  
│    └── processed/  
├── models/  
├── src/  
│    ├── data/  
│    │    ├── data_cleaning.py  
│    │    └── data_preparation.py  
│    ├── features/  
│    │    └── data_preprocessing.py  
│    └── models/  
│         ├── train.py  
│         └── evaluation.py  
└── notebooks/
```

## 🚀 Installation

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Manas-cmd467/swiggy.git](https://github.com/Manas-cmd467/swiggy.git)
   cd swiggy
   ```
2. **Create a virtual environment:**
   ```bash
   python -m venv .venv
   ```
3. **Activate the environment (Windows):**
   ```bash
   .venv\Scripts\activate
   ```
4. **Install dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

## 🏃‍♂️ Running the ML Pipeline

Run the complete pipeline with DVC to clean data, preprocess features, train the model, evaluate it, and save the artifacts:
```bash
dvc repro
```

## 🌐 Running the API

Start the FastAPI server:
```bash
uvicorn app:app --reload
```
* **Server runs at:** `http://127.0.0.1:8000`
* **Interactive API documentation:** `http://127.0.0.1:8000/docs`

## 📡 API Demonstration

**Endpoint:** `POST /predict`

**Example Input:**
```json
{
 "ID": "4607",
 "Delivery_person_ID": "INDRES13DEL02",
 "Delivery_person_Age": "37",
 "Delivery_person_Ratings": "4.9",
 "Restaurant_latitude": 22.745049,
 "Restaurant_longitude": 75.892471,
 "Delivery_location_latitude": 22.765049,
 "Delivery_location_longitude": 75.912471,
 "Order_Date": "19-03-2022",
 "Time_Orderd": "11:30",
 "Time_Order_picked": "11:45",
 "Weatherconditions": "Sunny",
 "Road_traffic_density": "High",
 "Vehicle_condition": 2,
 "Type_of_order": "Snack",
 "Type_of_vehicle": "motorcycle",
 "multiple_deliveries": "0",
 "Festival": "No",
 "City": "Urban"
}
```

**Example Output:**
```json
{
 "predicted_delivery_time_minutes": 18.7566
}
```

## 🛠️ Technologies Used

* **Python**
* **Scikit-learn**
* **LightGBM**
* **FastAPI**
* **DVC**
* **MLflow**
* **DagsHub**
* **Docker**

## 💡 Key Learnings

This project demonstrates:
* Building reproducible ML pipelines
* Ensemble modeling with stacking
* Data versioning with DVC
* Experiment tracking with MLflow
* API deployment with FastAPI
* Production-style ML project structure

## 👨‍💻 Author

**Manas Ranjan**
* GitHub: [@Manas-cmd467](https://github.com/Manas-cmd467)

---
*Note: Once you save this, push it to your repository using:*
```bash
git add README.md
git commit -m "Updated project documentation"
git push origin main
```
