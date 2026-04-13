import os
import joblib
import pandas as pd
import numpy as np
from flask import Flask, render_template, request, jsonify
from models.download_from_hf import download

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

# --- Application Startup ---

# Run the self-heal check *before* loading models
initialize_artifacts()

# Load models
try:
    pipe = joblib.load(PIPE_PATH)
    column_names = joblib.load(COLUMNS_PATH)
    print("Models loaded successfully.")
except Exception as e:
    print(f"\nFATAL: Error loading model artifacts: {e}")
    print("Files might be corrupt. Try deleting the 'models' directory and restarting.")
    exit(1) # Exit if loading fails

# Initialize Flask App
app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Extract features from the JSON request
        raw_features = [
            request.json["orbital-period"],
            request.json["transit-epoch"],
            request.json["transit-depth"],
            request.json["planet-radius"],
            request.json["semi-major-axis"],
            request.json["inclination"],
            request.json["equilibrium-temp"],
            request.json["insolation-flux"],
            request.json["impact-parameter"],
            request.json["radius-ratio"],
            request.json["stellar-density"],
            request.json["star-distance"],
            request.json["num-transits"],
        ]
        
        # Create DataFrame with correct column names
        df = pd.DataFrame([raw_features], columns=column_names)

        # Get prediction and probabilities
        pred = int(pipe.predict(df)[0])
        proba = pipe.predict_proba(df)[0]

        # Format probabilities for the response
        proba_dict = {
            reverse_mapping[i]: round(p, 3) for i, p in enumerate(proba)
        }

        # Send response
        return jsonify(
            {"prediction": reverse_mapping[pred], "probabilities": proba_dict}
        )

    except KeyError as e:
        print(f"Prediction Error: Missing key in request {e}")
        return jsonify({"error": f"Missing feature in request: {e}"}), 400
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)