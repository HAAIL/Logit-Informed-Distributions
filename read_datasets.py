import pandas as pd
import openml
from ucimlrepo import fetch_ucirepo
import os

os.makedirs("Data", exist_ok=True)

# -----------------------------
# UCI datasets 
# -----------------------------

uci_datasets = {
    "pima_diabetes": 34,
    "heart_disease": 45,
    "breast_cancer_diagnostic": 17,
    "breast_cancer_original": 15,
    "hepatitis": 46,
    "chronic_kidney_disease": 336,
    "mammographic_mass": 161,
    "cervical_cancer": 383,
    "adult": 2,
    "bank_marketing": 222,
    "credit_approval": 27,
    "german_credit": 144,
    "default_credit_card": 350,
    "abalone": 1,
    "ecoli": 39,
    "yeast": 110,
    "glass": 42,
    "seeds": 236,
    "wine_quality_red": 186,
    "wine_quality_white": 186,
    "dry_bean": 602,
    "rice": 545,
    "concrete_strength": 165,
    "energy_efficiency": 242,
    "airfoil_noise": 291,
    "combined_cycle_power": 294,
    "bike_sharing": 275,
    "forest_fires": 162,
    "communities_crime": 183
}

for name, dataset_id in uci_datasets.items():
    try:
        dataset = fetch_ucirepo(id=dataset_id)
        X = dataset.data.features
        y = dataset.data.targets
        df = pd.concat([X, y], axis=1)
        df.to_csv(f"Data/{name}.csv", index=False)
        print(f"Saved {name}")
    except Exception as e:
        print(f"Failed {name}: {e}")


# -----------------------------
# OpenML datasets
# -----------------------------

openml_datasets = [
    "bank-marketing",
    "adult",
    "credit-g",
    "diabetes",
]

for name in openml_datasets:
    try:
        dataset = openml.datasets.get_dataset(name)
        X, y, _, _ = dataset.get_data()
        df = pd.concat([X, y], axis=1)
        df.to_csv(f"Data/openml_{name}.csv", index=False)
        print(f"Saved {name}")
    except Exception as e:
        print(f"Failed {name}: {e}")
