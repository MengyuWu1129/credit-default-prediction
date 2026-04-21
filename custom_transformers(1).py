# custom_transformers.py
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

class TargetEncoder(BaseEstimator, TransformerMixin):
    """Mean target encoding for high-cardinality categorical features."""
    def __init__(self):
        self.global_ = None
        self.maps_ = {}

    def fit(self, X, y):
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.global_ = float(y.mean())
        self.maps_ = {}
        for col in X.columns:
            df_tmp = pd.DataFrame({"feature": X[col], "target": y})
            self.maps_[col] = df_tmp.groupby("feature")["target"].mean()
        return self

    def transform(self, X):
        X = pd.DataFrame(X).copy()
        for col in X.columns:
            X[col] = X[col].map(self.maps_[col]).fillna(self.global_)
        return X.values
