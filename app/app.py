from fastapi import FastAPI
from .schema.validate import UserInput
from models.download_from_hf import download

import pandas as pd
import os
from pathlib import Path

# --- Configuration ---
MODEL_DIR = "models"
PIPE_PATH = os.path.join(MODEL_DIR, "pipe.pkl")
COLUMNS_PATH = os.path.join(MODEL_DIR, "column_names.pkl")
reverse_mapping = {0: "FALSE POSITIVE", 1: "CANDIDATE", 2: "CONFIRMED"}

# --- Self-Heal Function ---
def initialize_artifacts():
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
        print("Model artifacts found. Loading...")