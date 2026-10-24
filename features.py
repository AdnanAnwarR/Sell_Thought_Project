"""Fitur & preprocessing: dipakai bersama oleh training dan aplikasi (menghindari training-serving skew)."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

T0 = pd.Timestamp("2018-11-08", tz="UTC")
CENSOR_CUTOFF = pd.Timestamp("2023-11-22", tz="UTC")  # data berakhir 2024-01-21, item terjual maks 60 hari setelah dibuat

CAT_COLS = ["product_category", "product_department", "product_brand", "dc"]
NUM_COLS = ["t_days", "month_sin", "month_cos", "dow", "log_price", "log_cost", "margin_pct", "margin_abs"]


class FeatureBuilder(BaseEstimator, TransformerMixin):
    """Membangun fitur dari kolom mentah. Hanya memakai informasi yang tersedia saat stok dibuat
    (sold_at TIDAK dipakai -> tidak ada data leakage). Brand langka digabung ke 'OTHER'."""
    def __init__(self, min_brand_count=50):
        self.min_brand_count = min_brand_count

    def fit(self, X, y=None):
        vc = X["product_brand"].fillna("UNKNOWN").value_counts()
        self.brands_ = set(vc[vc >= self.min_brand_count].index)
        return self

    def transform(self, X):
        d = pd.DataFrame(index=X.index)
        ts = pd.to_datetime(X["created_at"], utc=True, format="ISO8601")
        d["t_days"] = (ts - T0).dt.total_seconds() / 86400
        m = ts.dt.month
        d["month_sin"] = np.sin(2 * np.pi * m / 12)
        d["month_cos"] = np.cos(2 * np.pi * m / 12)
        d["dow"] = ts.dt.dayofweek
        price, cost = X["product_retail_price"].astype(float), X["cost"].astype(float)
        d["log_price"] = np.log1p(price)
        d["log_cost"] = np.log1p(cost)
        d["margin_abs"] = price - cost
        d["margin_pct"] = (price - cost) / price.replace(0, np.nan)
        d["product_category"] = X["product_category"].fillna("UNKNOWN")
        d["product_department"] = X["product_department"].fillna("UNKNOWN")
        b = X["product_brand"].fillna("UNKNOWN")
        d["product_brand"] = b.where(b.isin(self.brands_), "OTHER")
        d["dc"] = X["product_distribution_center_id"].astype(str)
        return d
