from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "model.joblib"

bundle = joblib.load(MODEL_PATH)

dv = bundle["vectorizer"]
model = bundle["model"]


app = FastAPI(
    title="Telco Customer Churn API",
    description="Bir müşterinin şirketten ayrılma olasılığını tahmin eder.",
    version="1.0.0"
)


class CustomerInput(BaseModel):
    gender: str
    seniorcitizen: int = Field(ge=0, le=1)
    partner: str
    dependents: str
    tenure: int = Field(ge=0)

    phoneservice: str
    multiplelines: str
    internetservice: str
    onlinesecurity: str
    onlinebackup: str
    deviceprotection: str
    techsupport: str
    streamingtv: str
    streamingmovies: str

    contract: str
    paperlessbilling: str
    paymentmethod: str

    monthlycharges: float = Field(ge=0)
    totalcharges: float = Field(ge=0)


@app.get("/")
def home():
    return {
        "message": "Customer Churn API çalışıyor.",
        "docs": "/docs"
    }


@app.get("/health")
def health():
    return {
        "status": "ok"
    }


@app.post("/predict")
def predict(customer: CustomerInput):
    try:
        customer_data = customer.model_dump()

        # Notebook'ta kategorik değerleri bu biçime çevirmiştin:
        # "Fiber optic" -> "fiber_optic"
        for column, value in customer_data.items():
            if isinstance(value, str):
                customer_data[column] = (
                    value
                    .lower()
                    .strip()
                    .replace(" ", "_")
                )

        # DictVectorizer tek müşteri bilgisini modele uygun sayısal
        # matrise dönüştürür.
        X = dv.transform([customer_data])

        churn_probability = float(
            model.predict_proba(X)[0, 1]
        )

        prediction = int(churn_probability >= 0.5)

        return {
            "prediction": prediction,
            "prediction_label": (
                "churn" if prediction == 1 else "no_churn"
            ),
            "churn_probability": round(churn_probability, 4),
            "no_churn_probability": round(
                1 - churn_probability,
                4
            )
        }

    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail=f"Tahmin yapılamadı: {error}"
        ) from error