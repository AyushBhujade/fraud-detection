from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import io
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Add this line temporarily
print("Static files path:", os.path.join(BASE_DIR, "frontend"))
print("Path exists:", os.path.exists(os.path.join(BASE_DIR, "frontend")))

# ✅ Load model ONCE at startup
with open("models/model.pkl", "rb") as f:
    model = pickle.load(f)

app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

@app.get("/", response_class=HTMLResponse)
def home():
    with open(os.path.join(BASE_DIR, "frontend", "index.html"), "r", encoding="utf-8") as f:
        return f.read()

@app.post("/predict-batch")
async def predict_batch(file: UploadFile = File(...)):
    contents = await file.read()
    df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
    df.dropna(inplace=True)
    print("Rows received:", len(df))
    predictions = model.predict(df)  # add column filtering if needed
    df["prediction"] = predictions
    return df.to_dict(orient="records")