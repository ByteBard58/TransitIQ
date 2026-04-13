from fastapi import FastAPI, Depends, UploadFile
from fastapi.requests import Request
from fastapi.responses import JSONResponse, FileResponse
from fastapi.exceptions import HTTPException
from fastapi.staticfiles import StaticFiles
from .schema.validate import UserInput
from models.download_from_hf import download
from sklearn.pipeline import Pipeline
from typing import Tuple, List

import joblib
import pandas as pd
import numpy as np
from contextlib import asynccontextmanager
import os
from pathlib import Path

# --- Configuration ---
MODEL_DIR = "models"
PIPE_PATH = os.path.join(MODEL_DIR, "pipe.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "column_names.pkl")
INDEX_PATH = Path("app","templates","index.html")
ABOUT_PATH = Path("app","templates","about.html")
reverse_mapping = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

# --- Self-Heal Function ---
def initialize_artifacts() -> Tuple[Pipeline,np.ndarray]:
    """
    Checks if model artifacts exist. If not, runs the training script.
    """
    # 1. Ensure the model directory exists
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # 2. Check for missing files
    pipe_exists = os.path.exists(PIPE_PATH)
    columns_exists = os.path.exists(COLUMNS_PATH)
    
    if not pipe_exists or not columns_exists:
        print("--- MODEL ARTIFACTS MISSING ---")
        if not pipe_exists:
            print(f"Missing: {PIPE_PATH}")
        if not columns_exists:
            print(f"Missing: {COLUMNS_PATH}")
        
        print("Downloading the saved models from Hugging Face... This may take a moment.")
        try:
            # Run the `download` function from `models/download_from_hf.py`
            download()
            print("Download complete. Artifacts generated successfully.")
            print("---------------------------------")
        except Exception as e:
            print(f"\nFATAL: Error during self-heal downloading: {e}")
            print("Application cannot start without model artifacts. Exitting......")
            exit(1) # Exit if training fails
    else:
        print("Model artifacts found.")

    print("Loading model artifacts....")
    pipe = joblib.load(PIPE_PATH)
    column_names = joblib.load(COLUMNS_PATH)
    print("Model artifacts are loaded. Ready for prediction 🚀")
    return pipe,column_names

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Loads the models at start
    """
    pipe,column_names = initialize_artifacts()

    app.state.pipe = pipe
    app.state.column_names = column_names

    yield

app = FastAPI(title="TransitIQ",version="3.0 (ByteBard58_Fork-FastAPI)",lifespan=lifespan)

# Mount static files
app.mount(name="static",path="/static",app=StaticFiles(
    directory=Path("app","static")
))

async def get_artifacts(request:Request) -> Tuple[Pipeline,np.ndarray]:
    """
    Helper to serve the artifacts in a route
    """
    return request.app.state.pipe, request.app.state.column_names

def validate_csv(target:pd.DataFrame,expected_columns:List) -> pd.DataFrame:
    """
    Helper for validating user-uploaded `.csv` files during batch prediction
    """
    if target.columns.to_list() != expected_columns:
        raise HTTPException(
            status_code=422,
            detail="The columns of the uploaded .csv file do not match with the expected list of columns or the order of them"

        )
    try:
        target.astype(float)
    except Exception:
        raise HTTPException(
            status_code=422,
            detail = "Provided values must be numeric (float-compatible)"
        )
    
    return target

@app.get("/health")
def health():
    msg = {
        "title":"TransitIQ",
        "version":"3.0(ByteBard58_Fork-FastAPI)",
        "status":"All systems operational"
    }
    return JSONResponse(content=msg,status_code=200)

@app.get("/")
def home():
    return FileResponse(INDEX_PATH)

@app.get("/about")
def about():
    return FileResponse(ABOUT_PATH,status_code=200)

reverse_mapping = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

@app.post("/predict")
def predict_with_manual_inputs(
    payload:UserInput, 
    dep:Tuple[Pipeline,np.ndarray] = Depends(get_artifacts)
):
    pipe, column_names = dep
    column_names:List = column_names.tolist()
    payload:dict = payload.model_dump(mode="json")

    sample = []
    for i,(key,val) in enumerate(payload.items()):
        if column_names[i] == key:
            sample.append(val)
        else:
            raise ValueError(f"Payload key {key} does not match expected column {column_names[i]}")
    sample = np.array(sample).reshape(1,-1)
    
    label = int(pipe.predict(sample)[0])
    proba:List = pipe.predict_proba(sample)[0].tolist()

    label:str = reverse_mapping.get(label)
    proba:dict = {cls:round(proba,3) for cls,proba in zip(reverse_mapping.values(),proba)}
    msg = {
        "status":"success",
        "prediction":label,
        "probabilities":proba
    }

    return JSONResponse(status_code=201,content=msg)

@app.post("/predict/batch")
async def predict_with_batch_input(
    file:UploadFile, 
    dep:Tuple[Pipeline,np.ndarray] = Depends(get_artifacts)
):
    pipe, column_names = dep
    column_names:List = column_names.tolist()

    ext = Path(file.filename).suffix
    if ext != ".csv":
        raise HTTPException(
            status_code=422, detail=f"Only .csv file is allowed as an upload, got {ext} instead"
        )
    
    try:
        df = pd.read_csv(file.file)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Failed to parse CSV file tracking: {str(e)}")
    df:np.ndarray = validate_csv(df,column_names).to_numpy()

    sample = df
    label:List[float] = pipe.predict(sample).tolist()
    proba:List[List[float]] = pipe.predict_proba(sample).tolist()

    label:List[str] = [reverse_mapping.get(l) for l in label]
    proba = [[round(value,3) for value in prob] for prob in proba]
    proba
    msg = {
        "status":"batch prediction successful",
        "predicted_labels":label,
        "predction_probability":proba
    }

    return JSONResponse(status_code=201,content=msg)