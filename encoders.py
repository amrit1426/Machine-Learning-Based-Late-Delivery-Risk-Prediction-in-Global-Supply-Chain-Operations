from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class SmoothedTargetEncoder(BaseEstimator, TransformerMixin):
    """
    M-estimate smoothed target encoder.
    Parameters
    ----------
    cols    : list of column names to encode
    m       : smoothing strength (default=20 is robust for most sizes)
    """
    def __init__(self, cols: list, m: float = 20.0):
        self.cols = cols
        self.m    = m

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.global_mean_ = float(y.mean())
        self.stats_        = {}
        for col in self.cols:
            if col not in X.columns:
                continue
            grp = y.groupby(X[col])
            n   = grp.count()
            s   = grp.sum()
            smoothed = (s + self.m * self.global_mean_) / (n + self.m)
            self.stats_[col] = smoothed.to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, mapping in self.stats_.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(self.global_mean_)
        return X