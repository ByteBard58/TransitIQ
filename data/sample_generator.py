"""
sample_generator.py

This script is used to generate samples directly from the main dataset. 
These samples are used to test the `/predict/batch` route. 
To run it, enter this in your command line:
```
python -m data.sample_generator
```
"""


import pandas as pd
import numpy as np


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


def create_test_sample(num_samples=10, random_seed=42):
    np.random.seed(random_seed)
    
    kepler_df = pd.read_csv("data/kepler_data.csv", comment="#")
    k2_df = pd.read_csv("data/k2_data.csv", comment="#")
    
    feature_list = [
        "koi_period", "koi_time0bk", "koi_depth", "koi_prad",
        "koi_sma", "koi_incl", "koi_teq", "koi_insol", "koi_impact",
        "koi_ror", "koi_srho", "koi_dor", "koi_num_transits"
    ]
    
    kepler_subset = kepler_df[feature_list]
    
    campaign_dates = {
        0: (2456725.0, 2456805.0), 1: (2456808.0, 2456891.0), 2: (2456893.0, 2456975.0),
        3: (2456976.0, 2457064.0), 4: (2457065.0, 2457159.0), 5: (2457159.0, 2457246.0),
        6: (2457250.0, 2457338.0), 7: (2457339.0, 2457420.0), 8: (2457421.0, 2457530.0),
        9: (2457504.0, 2457579.0), 10: (2457577.0, 2457653.0), 11: (2457657.0, 2457732.0),
        12: (2457731.0, 2457819.0), 13: (2457820.0, 2457900.0), 14: (2457898.0, 2457942.0),
        15: (2457941.0, 2458022.0), 16: (2458020.0, 2458074.0), 17: (2458074.0, 2458176.0),
        18: (2458151.0, 2458201.0), 19: (2458232.0, 2458348.0)
    }
    
    k2_df['campaigns'] = k2_df['k2_campaigns']
    k2_df[['obs_start_bjd', 'obs_end_bjd']] = k2_df['campaigns'].apply(
        lambda x: pd.Series(get_window(x, campaign_dates))
    )
    
    k2_df['n_min'] = np.ceil((k2_df['obs_start_bjd'] - k2_df['pl_tranmid']) / k2_df['pl_orbper'])
    k2_df['n_max'] = np.floor((k2_df['obs_end_bjd'] - k2_df['pl_tranmid']) / k2_df['pl_orbper'])
    k2_df['num_transits'] = (k2_df['n_max'] - k2_df['n_min'] + 1).clip(lower=0)
    
    k2_mapping = {
        "pl_orbper": "koi_period", "pl_tranmid": "koi_time0bk",
        "pl_trandep": "koi_depth", "pl_rade": "koi_prad", "pl_orbsmax": "koi_sma",
        "pl_orbincl": "koi_incl", "pl_eqt": "koi_teq", "pl_insol": "koi_insol",
        "pl_imppar": "koi_impact", "pl_ratror": "koi_ror", "pl_dens": "koi_srho",
        "pl_ratdor": "koi_dor", "num_transits": "koi_num_transits"
    }
    
    k2_subset = k2_df[list(k2_mapping.keys())].rename(columns=k2_mapping)
    
    combined = pd.concat([kepler_subset, k2_subset], ignore_index=True)
    
    combined = combined.dropna(subset=feature_list)
    
    sample_indices = np.random.choice(len(combined), size=min(num_samples, len(combined)), replace=False)
    sample = combined.iloc[sample_indices].copy()
    
    noise_factor = 0.15
    for col in feature_list:
        col_std = sample[col].std()
        if col_std > 0:
            noise = np.random.normal(0, col_std * noise_factor, size=len(sample))
            sample[col] = sample[col] + noise
            sample[col] = sample[col].clip(lower=0) if col in ["koi_depth", "koi_impact", "koi_ror", "koi_num_transits"] else sample[col]
    
    sample = sample[feature_list]
    
    output_path = "data/test_sample.csv"
    sample.to_csv(output_path, index=False)
    print(f"Created test sample with {len(sample)} rows at {output_path}")
    print(f"Columns: {sample.columns.tolist()}")
    print(f"\nSample preview:")
    print(sample.head())


if __name__ == "__main__":
    create_test_sample(num_samples=20)
