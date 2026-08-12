"""Valuation model: gradient-boosted trees on log(price), evaluated honestly.

Evaluation is 5-fold cross-validation where the sector encoding is re-learned
inside every training fold (no target leakage), and the model is compared
against the two pricing rules an agent would use without a model: global
median Rs/sqft x area, and sector-median Rs/sqft x area.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from homecast.features import build_features, sector_encoding, target

DEFAULT_PARAMS = {"n_estimators": 300, "max_depth": 3, "learning_rate": 0.06,
                  "subsample": 0.9, "random_state": 7}


def _metrics(actual_cr: np.ndarray, pred_cr: np.ndarray) -> dict:
    err = pred_cr - actual_cr
    return {
        "mae_lakh": float(np.mean(np.abs(err)) * 100),
        "mape_pct": float(np.mean(np.abs(err) / actual_cr) * 100),
        "r2": float(1 - np.sum(err ** 2) / np.sum((actual_cr - actual_cr.mean()) ** 2)),
    }


def _baseline_pred(train: pd.DataFrame, test: pd.DataFrame, by_sector: bool) -> np.ndarray:
    """Price = median Rs/sqft (global or sector) x area, in crore."""
    g = float(train["price_per_sqft"].median())
    if by_sector:
        m = train.groupby("sector")["price_per_sqft"].median()
        ppsf = test["sector"].map(m).fillna(g).to_numpy()
    else:
        ppsf = np.full(len(test), g)
    return ppsf * test["area"].to_numpy() / 1e7


def evaluate(df: pd.DataFrame, params: dict = DEFAULT_PARAMS, n_splits: int = 5) -> dict:
    df = df.reset_index(drop=True)
    oof = {k: np.zeros(len(df)) for k in ("model", "baseline_sector", "baseline_global")}
    for tr_idx, te_idx in KFold(n_splits, shuffle=True, random_state=7).split(df):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        smap = sector_encoding(tr)                      # fold-local: no leakage
        est = GradientBoostingRegressor(**params)
        est.fit(build_features(tr, smap), target(tr))
        oof["model"][te_idx] = np.exp(est.predict(build_features(te, smap)))
        oof["baseline_sector"][te_idx] = _baseline_pred(tr, te, by_sector=True)
        oof["baseline_global"][te_idx] = _baseline_pred(tr, te, by_sector=False)
    actual = df["price"].to_numpy(dtype=float)
    out = {k: _metrics(actual, v) for k, v in oof.items()}
    out["residuals_log"] = (np.log(oof["model"]) - np.log(actual)).tolist()
    out["n"] = len(df)
    out["params"] = dict(params)
    out["n_splits"] = n_splits
    return out


@dataclass(frozen=True)
class FittedModel:
    model: GradientBoostingRegressor
    sector_map: dict
    band: tuple
    ranges: dict
    metrics: dict = field(repr=False)


def train_final(df: pd.DataFrame, params: dict = DEFAULT_PARAMS) -> FittedModel:
    metrics = evaluate(df, params)
    smap = sector_encoding(df)
    est = GradientBoostingRegressor(**params)
    est.fit(build_features(df, smap), target(df))
    res = np.asarray(metrics["residuals_log"])
    band = (float(np.quantile(res, 0.10)), float(np.quantile(res, 0.90)))
    ranges = {c: [float(df[c].min()), float(df[c].max())]
              for c in ("area", "bedrooms", "bathrooms", "luxury_score")}
    return FittedModel(est, smap, band, ranges, metrics)


def predict_price(fitted: FittedModel, X: pd.DataFrame) -> np.ndarray:
    return np.exp(fitted.model.predict(X))
