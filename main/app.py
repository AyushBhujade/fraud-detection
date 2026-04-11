import time
import mlflow
import dagshub
import os
import io
import pandas as pd
from fastapi.responses import Response
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST, CollectorRegistry
from prometheus_client import Counter, Histogram






app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


REQUEST_COUNT = Counter(
    "http_requests_total",
    "Total HTTP Requests",
    ["method", "endpoint"]
)

REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds",
    "Request latency"
)

PREDICTION_COUNT = Counter(
    "model_predictions_total",
    "Total predictions"
)

FRAUD_COUNT = Counter(
    "fraud_predictions_total",
    "Fraud predictions"
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
dagshub_token=os.getenv("DAGSHUB_AUTH_TOKEN")
repo_name="fraud-detection"
repo_owner="ayushbhujade2005"
if dagshub_token:
    os.environ["MLFLOW_TRACKING_USERNAME"] = repo_owner
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token
    
else:
    raise ValueError("DAGSHUB_AUTH_TOKEN not found.")

# ✅ Set tracking URI directly
mlflow.set_tracking_uri(f"https://dagshub.com/{repo_owner}/{repo_name}.mlflow") 

# Add this line temporarily

# ✅ Load model ONCE at startup
def get_latest_model_version(model_name):
    client = mlflow.MlflowClient()
    # Prefer staging if it exists, then fall back to latest by version number.
    try:
        latest_version = client.get_latest_versions(model_name, stages=["Staging"])
        if latest_version:
            return latest_version[0].version
        
    except Exception:
        # If stages are disabled or the model isn't found yet, fall back below.
        pass

    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    latest = max(versions, key=lambda v: int(v.version))
    return latest.version


def load_registered_model(model_name):
    version = get_latest_model_version(model_name)
    if version is None:
        return None
    model_uri = f"models:/{model_name}/{version}"
    return mlflow.pyfunc.load_model(model_uri)

model = load_registered_model("new_XGBoost")

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(BASE_DIR, "frontend", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    start_time = time.time()
    REQUEST_COUNT.labels(method="POST", endpoint="/predict").inc()
    if model is None:
        return {"error": "Model not found in MLflow registry. Check MLFLOW_TRACKING_URI / MLFLOW_REGISTRY_URI."}
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df.dropna(inplace=True)
    print("Rows received:", len(df))
    predictions = model.predict(df)  # add column filtering if needed
    
    PREDICTION_COUNT.inc()
    
    fraud_count = (predictions == 1).sum()
    FRAUD_COUNT.inc(fraud_count)
    
    REQUEST_LATENCY.observe(time.time() - start_time)
    df["prediction"] = predictions
    return df.to_dict(orient="records")

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
