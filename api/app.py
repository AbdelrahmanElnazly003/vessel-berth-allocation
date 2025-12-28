from fastapi import FastAPI, UploadFile, File
import pandas as pd
import joblib
import numpy as np

model = joblib.load("model/logistic_regression_pipeline.pkl")

app = FastAPI(
    title="Vessel Berth Allocation API",
    description="Predict the most suitable berth for an incoming vessel",
    version="1.0"
)

@app.get("/")
def health_check():
    return {"status": "API is running"}

@app.post("/predict")
async def predict_berth(file: UploadFile = File(...)):

    df = pd.read_csv(file.file)

    if "BerthID" in df.columns:
        df = df.drop(columns=["BerthID"])

    top1_preds = model.predict(df)

    proba = model.predict_proba(df)
    classes = model.classes_

    top3_indices = np.argsort(proba, axis=1)[:, -3:]
    top3_preds = [[classes[i] for i in row] for row in top3_indices]

    results = []
    for i in range(len(df)):
        results.append({
            "row_id": i,
            "top_1_berth": top1_preds[i],
            "top_3_berths": top3_preds[i]
        })

    return {
        "num_records": len(df),
        "predictions": results
    }
