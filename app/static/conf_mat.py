"""
conf_mat.py

Script to generate confusion matrix png files
"""

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from models.download_from_hf import download
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
import os

MODEL_DIR = Path("models")
PIPE_PATH = Path("models","pipe.pkl")
COLUMNS_PATH = Path("models","column_names.pkl")
CONF_MAT_PATH = Path("app","static","materials","confusion_mat.png")

def initialize_artifacts() -> Pipeline:
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
        pipe = joblib.load(PIPE_PATH)
        print("Model artifacts are loaded. Ready for prediction 🚀")
        return pipe

def get_window(camps, campaign_dates):
    if pd.isna(camps) or not camps:
        return np.nan, np.nan

    camps = str(camps).split(',') if isinstance(camps, str) else camps
    starts, ends = [], []

    for c in camps:
        try:
            camp_num = int(c.strip())
            if camp_num in campaign_dates:
                start, end = campaign_dates[camp_num]
                starts.append(start)
                ends.append(end)
        except (ValueError, KeyError):
            continue

    return (min(starts) if starts else np.nan, max(ends) if ends else np.nan)


def load_and_prepare_data():
    # Load Kepler dataset
    df_raw = pd.read_csv("data/kepler_data.csv", comment="#")
    feature_list = [
        "koi_disposition", "koi_period", "koi_time0bk", "koi_depth", "koi_prad",
        "koi_sma", "koi_incl", "koi_teq", "koi_insol", "koi_impact",
        "koi_ror", "koi_srho", "koi_dor", "koi_num_transits"
    ]
    df_1 = df_raw[feature_list].copy()

    # Load K2 dataset
    df_2 = pd.read_csv("data/k2_data.csv", comment="#")

    # Define campaign windows
    campaign_dates = {
        0: (2456725.0, 2456805.0), 1: (2456808.0, 2456891.0), 2: (2456893.0, 2456975.0),
        3: (2456976.0, 2457064.0), 4: (2457065.0, 2457159.0), 5: (2457159.0, 2457246.0),
        6: (2457250.0, 2457338.0), 7: (2457339.0, 2457420.0), 8: (2457421.0, 2457530.0),
        9: (2457504.0, 2457579.0), 10: (2457577.0, 2457653.0), 11: (2457657.0, 2457732.0),
        12: (2457731.0, 2457819.0), 13: (2457820.0, 2457900.0), 14: (2457898.0, 2457942.0),
        15: (2457941.0, 2458022.0), 16: (2458020.0, 2458074.0), 17: (2458074.0, 2458176.0),
        18: (2458151.0, 2458201.0), 19: (2458232.0, 2458348.0)
    }

    # Add observation window
    df_2['campaigns'] = df_2['k2_campaigns']
    df_2[['obs_start_bjd', 'obs_end_bjd']] = df_2['campaigns'].apply(
        lambda x: pd.Series(get_window(x, campaign_dates))
    )

    # Transit counting
    df_2['n_min'] = np.ceil((df_2['obs_start_bjd'] - df_2['pl_tranmid']) / df_2['pl_orbper'])
    df_2['n_max'] = np.floor((df_2['obs_end_bjd'] - df_2['pl_tranmid']) / df_2['pl_orbper'])
    df_2['num_transits'] = (df_2['n_max'] - df_2['n_min'] + 1).clip(lower=0)

    # Select and rename columns
    df_2 = df_2[
        ["disposition", "pl_orbper", "pl_tranmid", "pl_trandep", "pl_rade",
         "pl_orbsmax", "pl_orbincl", "pl_eqt", "pl_insol", "pl_imppar",
         "pl_ratror", "pl_dens", "pl_ratdor", "num_transits"]
    ]

    mapping = {
        "disposition": "koi_disposition", "pl_orbper": "koi_period", "pl_tranmid": "koi_time0bk",
        "pl_trandep": "koi_depth", "pl_rade": "koi_prad", "pl_orbsmax": "koi_sma",
        "pl_orbincl": "koi_incl", "pl_eqt": "koi_teq", "pl_insol": "koi_insol",
        "pl_imppar": "koi_impact", "pl_ratror": "koi_ror", "pl_dens": "koi_srho",
        "pl_ratdor": "koi_dor", "num_transits": "koi_num_transits"
    }
    df_2 = df_2.rename(columns=mapping)

    # Combine both datasets
    df = pd.concat([df_1, df_2])

    # Prepare input/output
    X = df.iloc[:, 1:].to_numpy()
    y = df["koi_disposition"].map({
        "FALSE POSITIVE": 0, "CANDIDATE": 1, "CONFIRMED": 2, "REFUTED": 0
    }).to_numpy()

    return X, y, df.columns[1:]

def main() -> None:
    pipe = initialize_artifacts()
    x,y,column_names = load_and_prepare_data()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=1/3, shuffle=True, random_state=91, stratify=y
    )

    labels = ["FALSE POSITIVE","CANDIDATE","CONFIRMED"]
    y_true = y_test
    y_pred = pipe.predict(x_test)

    cm = confusion_matrix(y_true,y_pred)

    # Custom cosmic dark theme
    plt.style.use('dark_background')
    
    fig, ax = plt.subplots(figsize=(12, 12))
    fig.patch.set_facecolor('#0a0a0a')
    ax.set_facecolor('#0a0a0a')
    
    # Custom colormap - cosmic blue to cyan
    cmap = sns.color_palette([
        '#0a0a0a',
        '#001a2c',
        '#003355',
        '#004d7a',
        '#0066a3',
        '#0080cc',
        '#0099f5',
        '#00b3ff'
    ], as_cmap=True)
    
    # Create heatmap with custom styling
    sns.heatmap(
        cm,
        xticklabels=labels,
        yticklabels=labels,
        annot=True,
        fmt="d",
        square=True,
        cmap=cmap,
        cbar_kws={'label': 'Count', 'shrink': 1.0},
        linewidths=4,
        linecolor='#1a1a1a',
        ax=ax,
        annot_kws={'size': 32, 'weight': 'bold', 'color': '#ffffff'}
    )
    
    # Customize appearance
    ax.set_xlabel('Predicted', fontsize=22, color='#ffffff', fontweight='bold', labelpad=25)
    ax.set_ylabel('Actual', fontsize=22, color='#ffffff', fontweight='bold', labelpad=25)
    ax.set_title('Confusion Matrix', fontsize=36, color='#00BCFF', fontweight='bold', pad=40)
    
    # Style tick labels
    ax.tick_params(axis='both', colors='#a0a0a0', labelsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0, ha='center')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, ha='right')
    
    # Style colorbar
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.set_tick_params(color='#a0a0a0')
    cbar.outline.set_edgecolor('#1a1a1a')
    plt.setp(plt.getp(cbar.ax.axes, 'yticklabels'), color='#a0a0a0', size=16)
    
    plt.tight_layout()
    plt.savefig(CONF_MAT_PATH, dpi=300, facecolor='#0a0a0a', edgecolor='none', bbox_inches='tight', pad_inches=0.1)
    plt.close()
    print(f"Confusion Matrix is successfully saved at {str(CONF_MAT_PATH)}")
  
if __name__ == "__main__":
    main()