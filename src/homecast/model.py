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

# society_ppsf is a very strong feature (see feature_importances) -- strong
# enough that a model trained only on rows where it's informative leans on it
# far more than is safe for a real user, most of whom won't type in their
# building name. Without this, the model silently assumes the caller always
# knows their society; a user who doesn't got MAPE in the 30s, worse than the
# pre-Phase-2 model this whole upgrade was meant to beat. Masking a fraction
# of TRAINING rows to their sector-fallback value teaches the model to use
# area/sector/etc. instead of over-relying on society, so it degrades
# gracefully (a few points, not ~15) when the real prediction path can't
# supply one either. This is a robustness technique, not a hack -- do not
# "optimise" it away, and do not raise it toward 1.0 (that would just
# reproduce the old drop-society-entirely behaviour) or drop it to 0 (that
# reproduces the brittle, society-dependent model this fixes).
SOCIETY_MASK_FRACTION = 0.5
# Seed for the masking RNG, independent of KFold's random_state=7 shuffle but
# deliberately the same value, so a full run is reproducible end to end.
_SOCIETY_MASK_SEED = 7

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


def _mask_society(X: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Overwrite society_ppsf with the row's own sector_ppsf for a random
    ~SOCIETY_MASK_FRACTION of rows, so training sees the same "I don't know
    the society" situation a real caller is often in. Only ever applied to
    TRAINING features -- see the module docstring on SOCIETY_MASK_FRACTION."""
    X = X.copy()
    masked = rng.random(len(X)) < SOCIETY_MASK_FRACTION
    X.loc[masked, "society_ppsf"] = X.loc[masked, "sector_ppsf"]
    return X


def _without_society(X: pd.DataFrame) -> pd.DataFrame:
    """Simulate a caller who can't supply a society at predict time: replace
    every row's society_ppsf with its own sector_ppsf (the same fallback
    build_features already applies for an unrecognised/omitted society)."""
    X = X.copy()
    X["society_ppsf"] = X["sector_ppsf"]
    return X


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
    oof = {k: np.zeros(len(df)) for k in
           ("model", "model_no_society", "baseline_sector", "baseline_global")}
    mask_rng = np.random.default_rng(_SOCIETY_MASK_SEED)
    for tr_idx, te_idx in KFold(n_splits, shuffle=True, random_state=7).split(df):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        enc = fit_encoders(tr)                           # fold-local: no leakage
        Xtr = _mask_society(build_features(tr, enc), mask_rng)
        est = _make_estimator(model)
        est.fit(Xtr, target(tr))
        Xte = build_features(te, enc)
        oof["model"][te_idx] = np.exp(est.predict(Xte))
        # same out-of-fold rows, same trained (masking-robust) estimator, but
        # society withheld at predict time -- the number a real user who
        # doesn't know their building actually gets.
        oof["model_no_society"][te_idx] = np.exp(est.predict(_without_society(Xte)))
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
    Xfull = _mask_society(build_features(df, enc), np.random.default_rng(_SOCIETY_MASK_SEED))
    est.fit(Xfull, target(df))
    res = np.asarray(metrics["residuals_log"])
    band = (float(np.quantile(res, 0.10)), float(np.quantile(res, 0.90)))
    ranges = {c: [float(df[c].min()), float(df[c].max())]
              for c in ("area", "bedrooms", "bathrooms", "luxury_score")}
    return FittedModel(est, enc, band, ranges, metrics)


def predict_price(fitted: FittedModel, X: pd.DataFrame) -> np.ndarray:
    return np.exp(fitted.model.predict(X))
