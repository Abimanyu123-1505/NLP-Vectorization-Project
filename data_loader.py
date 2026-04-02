# ==============================
# 📥 DATA LOADER MODULE
# ==============================

import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd


def load_data():
    """
    Loads the IMDb Top 1000 dataset from Kaggle using kagglehub.
    
    Returns:
        df (pandas.DataFrame): Loaded dataset
    """
    try:
        df = kagglehub.dataset_load(
            KaggleDatasetAdapter.PANDAS,
            "harshitshankhdhar/imdb-dataset-of-top-1000-movies-and-tv-shows",
            "imdb_top_1000.csv"
        )

        print("✅ Dataset loaded successfully")
        print(f"Shape: {df.shape}")
        
        return df

    except Exception as e:
        print("❌ Error loading dataset:", e)
        return None


# ==============================
# 🔍 TEST RUN (optional)
# ==============================

if __name__ == "__main__":
    df = load_data()
    
    if df is not None:
        print("\n📊 First 5 rows:")
        print(df.head())