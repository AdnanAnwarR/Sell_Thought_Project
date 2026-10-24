import pandas as pd
from features import CENSOR_CUTOFF

def load_clean(path="inventory_items.csv"):
    df = pd.read_csv(path)
    df["created_at"] = pd.to_datetime(df["created_at"], utc=True, format="ISO8601")
    df["sold_at"] = pd.to_datetime(df["sold_at"], utc=True, format="ISO8601")
    df = df.drop_duplicates("id")
    df["is_sold"] = df["sold_at"].notna().astype(int)
    return df

def modelling_frame(df):
    """Buang item yang dibuat < 60 hari sebelum akhir data (belum sempat terjual = right-censored)."""
    return df[df["created_at"] < CENSOR_CUTOFF].copy()
