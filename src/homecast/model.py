"""Valuation model: gradient-boosted trees on log(price), evaluated honestly.

Evaluation is 5-fold cross-validation where every fold-local encoding is
re-learned inside every training fold (no target leakage), and the model is
compared against the two pricing rules an agent would use without a model:
global median Rs/sqft x area, and sector-median Rs/sqft x area.

Two named models are available (see ``MODELS``): ``"default"`` is a
GradientBoostingRegressor small enough to export to the browser dashboard;
``"accurate"`` is a heavier ExtraTreesRegressor for CLI-only use that beats
the default on offline metrics but has no staged/init_ structure and cannot
be exported (see ``homecast.export``).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

from homecast.features import Encoders, build_features, fit_encoders, target

DEFAULT_PARAMS = {"n_estimators": 500, "max_depth": 5, "learning_rate": 0.05,
                  "subsample": 0.9, "random_state": 7}
ACCURATE_PARAMS = {"n_estimators": 100, "random_state": 7, "n_jobs": -1}

# Named model registry. Each entry is a template estimator; every fold/final
# fit clones it fresh so no state leaks between folds or between cities.
MODELS: dict[str, BaseEstimator] = {
    "default": GradientBoostingRegressor(**DEFAULT_PARAMS),
    "accurate": ExtraTreesRegressor(**ACCURATE_PARAMS),
}


def _make_estimator(model: str) -> BaseEstimator:
    try:
        template = MODELS[model]
    except KeyError:
        raise ValueError(f"Unknown model '{model}'. Valid: "
                         f"{', '.join(sorted(MODELS))}") from None
    return clone(template)


def _validate_prices(df: pd.DataFrame) -> None:
    """Prices must be strictly positive: log(price) is the model target and
    price is the denominator of MAPE."""
    price = df["price"].to_numpy(dtype=float)
    if not np.all(np.isfinite(price) & (price > 0)):
        bad = int((~(np.isfinite(price) & (price > 0))).sum())
        raise ValueError(f"{bad} listing(s) have a non-positive or non-finite "
                         f"price; cannot evaluate in log space")


def _metrics(actual_cr: np.ndarray, pred_cr: np.ndarray) -> dict:
    err = pred_cr - actual_cr
    denom = float(np.sum((actual_cr - actual_cr.mean()) ** 2))
    if denom == 0:
        raise ValueError("cannot compute R2: all actual prices are identical")
    return {
        "mae_lakh": float(np.mean(np.abs(err)) * 100),
        "mape_pct": float(np.mean(np.abs(err) / actual_cr) * 100),
        "r2": float(1 - np.sum(err ** 2) / denom),
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


def evaluate(df: pd.DataFrame, model: str = "default", n_splits: int = 5) -> dict:
    df = df.reset_index(drop=True)
    _validate_prices(df)
    oof = {k: np.zeros(len(df)) for k in ("model", "baseline_sector", "baseline_global")}
    for tr_idx, te_idx in KFold(n_splits, shuffle=True, random_state=7).split(df):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        enc = fit_encoders(tr)                           # fold-local: no leakage
        est = _make_estimator(model)
        est.fit(build_features(tr, enc), target(tr))
        oof["model"][te_idx] = np.exp(est.predict(build_features(te, enc)))
        oof["baseline_sector"][te_idx] = _baseline_pred(tr, te, by_sector=True)
        oof["baseline_global"][te_idx] = _baseline_pred(tr, te, by_sector=False)
    actual = df["price"].to_numpy(dtype=float)
    out = {k: _metrics(actual, v) for k, v in oof.items()}
    out["residuals_log"] = (np.log(oof["model"]) - np.log(actual)).tolist()
    out["n"] = len(df)
    out["model_name"] = model
    out["params"] = MODELS[model].get_params()
    out["n_splits"] = n_splits
    return out


@dataclass(frozen=True)
class FittedModel:
    model: BaseEstimator
    encoders: Encoders
    band: tuple
    ranges: dict
    metrics: dict = field(repr=False)


def train_final(df: pd.DataFrame, model: str = "default") -> FittedModel:
    metrics = evaluate(df, model=model)
    enc = fit_encoders(df)
    est = _make_estimator(model)
    est.fit(build_features(df, enc), target(df))
    res = np.asarray(metrics["residuals_log"])
    band = (float(np.quantile(res, 0.10)), float(np.quantile(res, 0.90)))
    ranges = {c: [float(df[c].min()), float(df[c].max())]
              for c in ("area", "bedrooms", "bathrooms", "luxury_score")}
    return FittedModel(est, enc, band, ranges, metrics)


def predict_price(fitted: FittedModel, X: pd.DataFrame) -> np.ndarray:
    return np.exp(fitted.model.predict(X))
