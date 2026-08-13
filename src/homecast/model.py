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

from homecast.features import (Encoders, FEATURE_COLUMNS, build_features,
                               feature_columns_for, fit_encoders, target)

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
    TRAINING features -- see the module docstring on SOCIETY_MASK_FRACTION.

    A city whose feed has no building names has no society_ppsf feature at
    all (see homecast.features), so there is nothing to mask and nothing to
    be over-reliant on: this is a no-op there. The RNG is deliberately still
    NOT drawn from in that case -- consuming a draw the masking never uses
    would make one city's stream depend on another's schema."""
    if "society_ppsf" not in X.columns:
        return X
    X = X.copy()
    masked = rng.random(len(X)) < SOCIETY_MASK_FRACTION
    X.loc[masked, "society_ppsf"] = X.loc[masked, "sector_ppsf"]
    return X


def _without_society(X: pd.DataFrame) -> pd.DataFrame:
    """Simulate a caller who can't supply a society at predict time: replace
    every row's society_ppsf with its own sector_ppsf (the same fallback
    build_features already applies for an unrecognised/omitted society).

    For a city with no society feature this is a no-op, so the reported
    "without a society given" metric is identical to the headline one --
    correctly so: there is no society to withhold."""
    if "society_ppsf" not in X.columns:
        return X
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


def model_beats_baseline(metrics: dict) -> bool:
    """Whether the learned model's headline MAPE is a genuine win over this
    city's locality-median (sector) rule of thumb -- the same "agent quotes
    Rs/sq.ft." rule a buyer could apply with no model at all.

    A TIE does not count as beating it: "beats" means a real win, not
    "at least as good as". A model that only ties or loses has no business
    being presented as the estimate (see baseline_served in evaluate() and
    baseline_served_reason below) -- this is the one place that decision is
    made, so every caller (CLI, dashboard export, tests) reads the same
    verdict instead of each re-deriving its own threshold.

    Takes a plain metrics dict (not a FittedModel) on purpose: this is pure
    and directly testable against a synthetic {"model": ..., "baseline_sector":
    ...} dict, without needing to fit an estimator to prove the rule itself.
    """
    return metrics["model"]["mape_pct"] < metrics["baseline_sector"]["mape_pct"]


# Catalogue features beyond the always-present core (area, bedrooms, and the
# locality encodings -- see features.CORE_FEATURE_COLUMNS) that could give a
# model real signal a locality-median rule cannot already capture on its own.
# Listed here, once, with the plain-English word each reads as in a sentence,
# so baseline_served_reason can explain what a losing city's feed is missing
# without hardcoding that explanation per city.
_EXPLANATORY_FEATURES = (
    ("bathrooms", "bathrooms"), ("society_ppsf", "society"),
    ("furnishing_code", "furnishing"), ("age_code", "age"),
    ("balcony_code", "balcony count"), ("is_house", "property type"),
    ("luxury_score", "amenity score"), ("amenity_count", "amenity count"),
    ("is_resale", "resale status"),
)


# Below this MAPE, neither the model nor the locality-median rule of thumb
# is good enough to present as a price -- a confident-looking number this far
# off a listing's actual price is worse than admitting the market can't be
# priced responsibly with the data on hand (see serving_status below). 30%
# is a deliberately round, plainly-justifiable line: it is roughly double the
# ~13-16% irreducible noise floor this project's own data shows even a
# well-served city (bathrooms, society, furnishing, age all present) cannot
# get below, so a city sitting at or above 30% is not "a bit worse than
# usual" -- it is missing real signal, not just noise. It is set ONCE, here,
# as a general rule for every city, and must NOT be tuned per city to change
# which cities pass or fail it -- that would defeat the entire point of
# having a floor.
QUALITY_FLOOR_MAPE_PCT = 30.0


def meets_quality_floor(metrics: dict) -> bool:
    """Whether the BETTER of {model, locality-median rule} clears
    QUALITY_FLOOR_MAPE_PCT for this city.

    Deliberately independent of model_beats_baseline: that function decides
    WHICH of the two numbers is better; this one asks whether even the
    better one is good enough to show anyone at all. A city can lose to its
    baseline and still clear the floor (baseline-served), or win against its
    baseline and still fail to clear it (e.g. Mumbai: the model technically
    beats a hopeless baseline, but neither is within 30% MAPE of the truth).
    """
    better_mape = (metrics["model"]["mape_pct"] if model_beats_baseline(metrics)
                  else metrics["baseline_sector"]["mape_pct"])
    return better_mape < QUALITY_FLOOR_MAPE_PCT


def serving_status(metrics: dict) -> str:
    """The three-way serving decision for a city, from its metrics alone.

    "model"      -- the learned model beats the baseline AND clears the floor.
    "baseline"   -- the baseline is the better number AND clears the floor.
    "not_served" -- neither clears the floor: even the better of the two is
      too unreliable to present as a price at all. Serving a confident number
      at that error rate is worse than serving nothing (see the module
      docstring's QUALITY_FLOOR_MAPE_PCT comment) -- a dashboard/CLI reading
      this value must not show a headline price estimate when it is
      "not_served", only the measured error rates and why.

    General on purpose, exactly like model_beats_baseline: derived only from
    THIS city's own metrics dict, never its name or key, so any current or
    future city gets the same mechanical decision.
    """
    if not meets_quality_floor(metrics):
        return "not_served"
    return "model" if model_beats_baseline(metrics) else "baseline"


def _missing_signal_note(columns) -> str:
    """Clause naming which explanatory columns (see _EXPLANATORY_FEATURES)
    this city's feed lacks -- or a fixed clause when it has all of them.
    Shared by baseline_served_reason and not_served_reason so "what is this
    market's feed missing" is computed in exactly one place, from THIS
    city's own feature list, never hardcoded per city.
    """
    cols = set(columns)
    have = [label for f, label in _EXPLANATORY_FEATURES if f in cols]
    missing = [label for f, label in _EXPLANATORY_FEATURES if f not in cols]
    if not missing:
        return "even with the full feature set available for this city"
    signal = "area, bedrooms and locality" + (f", {', '.join(have)}" if have else "")
    return f"the available columns are only {signal} (no {', '.join(missing)})"


def baseline_served_reason(metrics: dict, columns) -> str:
    """Plain-language explanation of why a city's estimate falls back to the
    locality-median rule of thumb instead of the learned model.

    General on purpose, like model_beats_baseline above: derived from this
    city's OWN metrics and its OWN feature list (never its name or key), so a
    future city that also loses to its baseline gets an accurate, specific
    reason automatically instead of the four cities checked at the time this
    was written getting hand-written prose that silently goes stale for a
    fifth.
    """
    m, b = metrics["model"]["mape_pct"], metrics["baseline_sector"]["mape_pct"]
    verdict = "tied" if abs(m - b) < 1e-9 else "lost to"
    reason = (f"the learned model {verdict} the ₹/sq.ft. rule of thumb "
             f"for this market ({m:.1f}% MAPE vs {b:.1f}% for the rule)")
    note = _missing_signal_note(columns)
    sep = " " if note.startswith("even") else " -- "
    return reason + sep + note


def not_served_reason(metrics: dict, columns) -> str:
    """Plain-language explanation of why NEITHER number is presented for this
    city: the data available is not good enough to price a property
    responsibly.

    Deliberately does NOT reuse baseline_served_reason's "lost to"/"tied"
    framing: a not-served city can have EITHER verdict underneath (Mumbai's
    model narrowly beats its own hopeless baseline and is still not_served,
    Delhi NCR's model loses AND is not_served), so a sentence asserting one
    direction would be wrong for the other. Instead this states both MAPE
    numbers plainly against the floor itself. Reuses _missing_signal_note
    (a thin feature set is usually WHY a market is this hard to price); the
    numbers themselves are "how dispersed the market is" -- how much of a
    listing's own price this data alone cannot explain, even granting it the
    correct locality and area.
    """
    m, b = metrics["model"]["mape_pct"], metrics["baseline_sector"]["mape_pct"]
    better = min(m, b)
    return (f"the data available for this market is not good enough to "
            f"price a property responsibly: neither the learned model "
            f"({m:.1f}% MAPE) nor the ₹/sq.ft. rule of thumb ({b:.1f}% MAPE) "
            f"clears the {QUALITY_FLOOR_MAPE_PCT:.0f}% MAPE quality floor "
            f"(best of the two is {better:.1f}%) -- {_missing_signal_note(columns)}")


def evaluate(df: pd.DataFrame, model: str = "default", n_splits: int = 5,
             columns: list[str] | None = None) -> dict:
    df = df.reset_index(drop=True)
    _validate_prices(df)
    # Resolved ONCE, from the whole frame, then reused for every fold: a
    # fold-local re-inference could pick a different feature set per fold.
    # This is schema-only (which columns exist), never target-derived, so
    # deriving it from all rows leaks nothing -- the encodings that DO see
    # the target are still re-fit inside each training fold below.
    if columns is None:
        columns = feature_columns_for(df)
    oof = {k: np.zeros(len(df)) for k in
           ("model", "model_no_society", "baseline_sector", "baseline_global")}
    mask_rng = np.random.default_rng(_SOCIETY_MASK_SEED)
    for tr_idx, te_idx in KFold(n_splits, shuffle=True, random_state=7).split(df):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        enc = fit_encoders(tr)                           # fold-local: no leakage
        Xtr = _mask_society(build_features(tr, enc, columns), mask_rng)
        est = _make_estimator(model)
        est.fit(Xtr, target(tr))
        Xte = build_features(te, enc, columns)
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
    # Same out-of-fold construction as the model's own residuals, for the
    # locality-median rule instead -- so a caller can build an honest
    # confidence band around the RULE's price when the rule, not the model,
    # is what gets shown (see baseline_served below and FittedModel.baseline_band).
    out["baseline_residuals_log"] = (np.log(oof["baseline_sector"]) - np.log(actual)).tolist()
    out["n"] = len(df)
    out["model_name"] = model
    out["params"] = MODELS[model].get_params()
    out["n_splits"] = n_splits
    out["columns"] = list(columns)
    # Whether this city's page/CLI output should present the model's own
    # number, or fall back to presenting the locality-median rule of thumb as
    # THE estimate (see model_beats_baseline / baseline_served_reason). Computed
    # here, once, from this same evaluate() run, so every consumer (CLI,
    # export payload, metrics.json) reads one shared answer instead of each
    # re-deriving its own opinion about what "beats" means.
    out["baseline_served"] = not model_beats_baseline(out)
    # The three-way decision (see serving_status): "model" / "baseline" /
    # "not_served". Independent of, and computed after, baseline_served
    # above -- a city can be "not model_beats_baseline" (baseline_served
    # True) yet still clear the quality floor (baseline-served), or can beat
    # its own baseline (baseline_served False) and still fail to clear the
    # floor (not_served, e.g. Mumbai). Every consumer that needs to know
    # whether to show a headline price at all should read THIS field, not
    # baseline_served, which only ever answers "model vs baseline", never
    # "is either one good enough".
    out["serving_status"] = serving_status(out)
    return out


# Numeric inputs a caller types in, so a query outside the training range can
# be refused (see valuation._check_range). Only those the city actually has
# end up in FittedModel.ranges.
RANGE_COLUMNS = ("area", "bedrooms", "bathrooms", "luxury_score")


@dataclass(frozen=True)
class FittedModel:
    model: BaseEstimator
    encoders: Encoders
    band: tuple
    ranges: dict
    metrics: dict = field(repr=False)
    # The exact feature list this estimator was fit on, in order. Carried on
    # the fitted model rather than re-inferred at predict time so a query
    # frame can never produce a different set (see features.build_features).
    # Defaults to the Gurgaon 13 so a model pickled before this field existed
    # still loads.
    columns: tuple[str, ...] = tuple(FEATURE_COLUMNS)
    # (q10, q90) of the locality-median rule's own OOF log residuals -- the
    # same construction as `band` above, but for the rule instead of the
    # model. Exists so a caller can put an honest range around the RULE's
    # price when a city is baseline_served and the rule, not the model, is
    # being presented as the estimate (see valuation.estimate). Defaults to
    # (0.0, 0.0) so a model pickled before this field existed still loads;
    # that default is never reached for a freshly trained model, which always
    # gets a real band computed below.
    baseline_band: tuple = (0.0, 0.0)


def train_final(df: pd.DataFrame, model: str = "default") -> FittedModel:
    columns = feature_columns_for(df)
    metrics = evaluate(df, model=model, columns=columns)
    enc = fit_encoders(df)
    est = _make_estimator(model)
    Xfull = _mask_society(build_features(df, enc, columns),
                          np.random.default_rng(_SOCIETY_MASK_SEED))
    est.fit(Xfull, target(df))
    res = np.asarray(metrics["residuals_log"])
    band = (float(np.quantile(res, 0.10)), float(np.quantile(res, 0.90)))
    baseline_res = np.asarray(metrics["baseline_residuals_log"])
    baseline_band = (float(np.quantile(baseline_res, 0.10)),
                     float(np.quantile(baseline_res, 0.90)))
    ranges = {c: [float(df[c].min()), float(df[c].max())]
              for c in RANGE_COLUMNS if c in df.columns}
    return FittedModel(est, enc, band, ranges, metrics, tuple(columns), baseline_band)


def predict_price(fitted: FittedModel, X: pd.DataFrame) -> np.ndarray:
    return np.exp(fitted.model.predict(X))
