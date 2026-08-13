import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.model_selection import KFold

from homecast.features import build_features, fit_encoders, target
from homecast.model import (DEFAULT_PARAMS, MODELS, SOCIETY_MASK_FRACTION,
                            _without_society, evaluate, predict_price,
                            train_final)


def test_evaluate_reports_model_and_baselines(clean_fixture):
    r = evaluate(clean_fixture, n_splits=3)
    for k in ("model", "model_no_society", "baseline_sector", "baseline_global"):
        assert set(r[k]) == {"mae_lakh", "mape_pct", "r2"}
        assert r[k]["mae_lakh"] > 0
    assert len(r["residuals_log"]) == len(clean_fixture)

def test_train_final_predicts_reasonably(clean_fixture):
    f = train_final(clean_fixture)
    X = build_features(clean_fixture, f.encoders)
    pred = predict_price(f, X)
    assert pred.shape == (len(clean_fixture),)
    assert (pred > 0).all()
    # in-sample fit on tiny data should at least beat predicting the median
    med = np.median(clean_fixture["price"])
    assert np.mean(np.abs(pred - clean_fixture["price"])) < np.mean(
        np.abs(med - clean_fixture["price"]))

def test_band_covers_about_eighty_percent_of_listings(clean_fixture):
    """The band must be a usable 80% interval, applied with the right sign.

    ``band`` holds the 10th/90th percentile of residual = log(pred) - log(actual),
    so actual = pred * exp(-residual): the q90 residual gives the LOW end of the
    true price and q10 gives the HIGH end. Reconstruct the out-of-fold
    predictions from the stored residuals and count how many actual prices land
    inside the interval. By construction that fraction is 0.80 on the data the
    band was fitted from, but the assertion still catches a band built from the
    wrong quantiles or from the wrong residual definition. The tolerance is wide
    (0.60-0.95) because the fixture is ~40 rows, where a single listing moves
    coverage by 2.4 points and quantile interpolation is coarse.
    """
    f = train_final(clean_fixture)
    q10, q90 = f.band
    assert q10 < q90                       # 10th percentile below the 90th
    actual = clean_fixture["price"].to_numpy(dtype=float)
    res = np.asarray(f.metrics["residuals_log"])
    pred = actual * np.exp(res)            # residual = log(pred) - log(actual)
    inside = (actual >= pred * np.exp(-q90)) & (actual <= pred * np.exp(-q10))
    coverage = float(inside.mean())
    assert 0.60 <= coverage <= 0.95, f"band covers {coverage:.0%}, expected ~80%"

def test_ranges_cover_data(clean_fixture):
    f = train_final(clean_fixture)
    assert f.ranges["area"][0] == clean_fixture["area"].min()
    assert f.ranges["area"][1] == clean_fixture["area"].max()

def test_zero_price_rejected(clean_fixture):
    bad = clean_fixture.copy()
    bad.loc[bad.index[0], "price"] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        evaluate(bad, n_splits=3)

def test_constant_price_r2_rejected(clean_fixture):
    flat = clean_fixture.copy()
    flat["price"] = 1.0
    with pytest.raises(ValueError, match="identical"):
        evaluate(flat, n_splits=3)


# ── model registry ─────────────────────────────────────────────────────────

def test_models_registry_has_default_and_accurate():
    assert set(MODELS) == {"default", "accurate"}
    assert isinstance(MODELS["default"], GradientBoostingRegressor)
    assert isinstance(MODELS["accurate"], ExtraTreesRegressor)

def test_unknown_model_name_rejected(clean_fixture):
    with pytest.raises(ValueError, match="Unknown model"):
        evaluate(clean_fixture, model="fastest", n_splits=3)

@pytest.fixture()
def signal_frame() -> pd.DataFrame:
    """Big enough, with real signal beyond sector (area and luxury both move
    price), that a fitted model has something to learn past the naive
    per-sector median rule -- unlike the tiny 41-row `clean_fixture`, whose
    price-per-sqft is pure noise independent of sector by construction and so
    isn't a fair test of "does the model beat the baseline"."""
    rng = np.random.default_rng(5)
    n = 400
    sector_names = [f"sector {i}" for i in range(8)]
    sectors = rng.choice(sector_names, n)
    area = rng.uniform(600.0, 3000.0, n)
    luxury = rng.uniform(0.0, 150.0, n)
    base_ppsf = {s: rng.uniform(5000.0, 20000.0) for s in sector_names}
    noise = rng.normal(0.0, 0.05, n)
    ppsf = np.array([base_ppsf[s] for s in sectors]) * (1 + luxury / 500) * np.exp(noise)
    return pd.DataFrame({
        "sector": sectors,
        "society": [f"society {i % 40}" for i in range(n)],
        "property_type": rng.choice(["flat", "house"], n, p=[0.8, 0.2]),
        "price": area * ppsf / 1e7,
        "price_per_sqft": ppsf,
        "area": area,
        "bedrooms": rng.integers(1, 6, n),
        "bathrooms": rng.integers(1, 6, n),
        "balcony": rng.choice(["0", "1", "2", "3", "3+"], n),
        "furnishing_type": rng.choice(
            ["unfurnished", "semi-furnished", "furnished"], n),
        "luxury_score": luxury,
        "age_possession": "New Property",
    })

def test_default_and_accurate_both_beat_sector_baseline(signal_frame):
    for name in ("default", "accurate"):
        r = evaluate(signal_frame, model=name, n_splits=5)
        assert r["model"]["mae_lakh"] < r["baseline_sector"]["mae_lakh"], name

def test_train_final_accurate_model_fits_and_predicts(clean_fixture):
    f = train_final(clean_fixture, model="accurate")
    assert isinstance(f.model, ExtraTreesRegressor)
    X = build_features(clean_fixture, f.encoders)
    pred = predict_price(f, X)
    assert (pred > 0).all()

def test_evaluate_does_not_mutate_shared_registry_state(clean_fixture):
    """Each fold/final fit must clone its estimator template -- otherwise
    successive evaluate() calls would silently reuse a stale fitted model."""
    before = MODELS["default"].get_params()
    evaluate(clean_fixture, n_splits=3)
    assert MODELS["default"].get_params() == before
    assert not hasattr(MODELS["default"], "estimators_")


# ── robustness when the caller doesn't know the society ────────────────────
# society_ppsf is a very strong feature (see the module docstring on
# SOCIETY_MASK_FRACTION); a model trained without ever seeing a masked
# society learns to lean on it far too hard, so real users who don't type in
# their building name get a MUCH worse number than the reported CV metric --
# the exact gap this masking fixes.

@pytest.fixture()
def society_signal_frame() -> pd.DataFrame:
    """Sector explains most of the price; society adds real signal on top
    (not noise) -- enough that an UNPROTECTED model would still lean on it
    heavily, so this is a meaningful test of graceful degradation, not a
    trivial one where withholding society costs nothing to begin with."""
    rng = np.random.default_rng(17)
    n = 900
    sector_names = [f"sector {i}" for i in range(10)]
    sectors = rng.choice(sector_names, n)
    societies = rng.choice([f"society {i}" for i in range(60)], n)
    sector_base = {s: rng.uniform(6000.0, 20000.0) for s in sector_names}
    society_mult = {so: float(np.exp(rng.normal(0.0, 0.25)))
                    for so in [f"society {i}" for i in range(60)]}
    noise = rng.normal(0.0, 0.05, n)
    ppsf = (np.array([sector_base[s] for s in sectors])
            * np.array([society_mult[so] for so in societies])
            * np.exp(noise))
    area = rng.uniform(700.0, 2500.0, n)
    return pd.DataFrame({
        "sector": sectors, "society": societies, "property_type": "flat",
        "price": area * ppsf / 1e7, "price_per_sqft": ppsf, "area": area,
        "bedrooms": rng.integers(1, 6, n), "bathrooms": rng.integers(1, 6, n),
        "balcony": rng.choice(["0", "1", "2", "3", "3+"], n),
        "furnishing_type": rng.choice(
            ["unfurnished", "semi-furnished", "furnished"], n),
        "luxury_score": rng.uniform(0.0, 150.0, n),
        "age_possession": "New Property",
    })

def _fold_gap_with_and_without_masking(df: pd.DataFrame, n_splits: int = 5) -> tuple[float, float]:
    """Reproduce evaluate()'s fold loop twice on the identical folds/model --
    once through the real (masked-training) code path, once through a local
    unmasked variant -- and return (masked_gap, unmasked_gap), each the
    MAPE-without-society minus MAPE-with-society. Only used to prove masking
    is what closes the gap; never call this outside this test."""
    masked = evaluate(df, n_splits=n_splits)
    masked_gap = masked["model_no_society"]["mape_pct"] - masked["model"]["mape_pct"]

    df = df.reset_index(drop=True)
    import homecast.model as model_mod
    oof, oof_no_soc = np.zeros(len(df)), np.zeros(len(df))
    for tr_idx, te_idx in KFold(n_splits, shuffle=True, random_state=7).split(df):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        enc = fit_encoders(tr)
        est = GradientBoostingRegressor(**DEFAULT_PARAMS)
        est.fit(build_features(tr, enc), target(tr))         # <- no masking
        Xte = build_features(te, enc)
        oof[te_idx] = np.exp(est.predict(Xte))
        oof_no_soc[te_idx] = np.exp(est.predict(model_mod._without_society(Xte)))
    actual = df["price"].to_numpy(dtype=float)
    mape_known = float(np.mean(np.abs(oof - actual) / actual) * 100)
    mape_unknown = float(np.mean(np.abs(oof_no_soc - actual) / actual) * 100)
    return masked_gap, mape_unknown - mape_known


def test_model_degrades_gracefully_without_society(society_signal_frame):
    """With masked training, withholding society at predict time must cost
    meaningfully less than an equivalent model trained without masking, on
    the identical folds/data -- proof the masking mechanism (not just luck on
    one fixture) is what limits the damage."""
    assert 0.0 < SOCIETY_MASK_FRACTION < 1.0, (
        "masking must be genuinely partial -- 0 reproduces the brittle "
        "society-dependent model, 1 reproduces drop-society-entirely")
    masked_gap, unmasked_gap = _fold_gap_with_and_without_masking(society_signal_frame)
    assert masked_gap >= -1e-9, "withholding real information should not IMPROVE the estimate"
    assert masked_gap < unmasked_gap - 2.0, (
        f"masked-training gap ({masked_gap:.2f}pp) is not meaningfully "
        f"smaller than the unmasked gap ({unmasked_gap:.2f}pp) -- masking "
        f"doesn't look like it's actually protecting predictions")

def test_model_degrades_gracefully_without_society_on_real_data():
    """The concrete numeric claim, on the actual committed Gurgaon dataset:
    ~1.6pp with masking (this test), vs. ~15pp measured without it. A ceiling
    of 3pp (the coordinator's own "~3pp" bound) fails hard on a regression to
    the old behaviour while leaving headroom for legitimate retraining
    variance."""
    df = pd.read_csv("data/gurgaon/processed/listings_clean.csv")
    r = evaluate(df, n_splits=5)
    known, unknown = r["model"]["mape_pct"], r["model_no_society"]["mape_pct"]
    gap = unknown - known
    assert gap >= -1e-9
    assert gap <= 3.0, (
        f"MAPE without society ({unknown:.2f}%) is {gap:.2f}pp worse than "
        f"with it ({known:.2f}%) on the real dataset -- expected ~1.6pp")


def test_train_final_masks_society_in_the_shipped_final_fit():
    """`train_final` has its OWN society-masking call (`Xfull = _mask_society(...)`)
    separate from `evaluate`'s fold loop -- masking could be silently dropped
    from just that one line and every other test in this file would still
    pass, because they all exercise `evaluate`'s fold loop, not the final fit
    that actually becomes `model.joblib`/`model.json`/the browser export.
    `metrics.json` would keep advertising the masked, graceful-degradation
    numbers (it comes from `evaluate`, untouched by such a regression) while
    the shipped model itself quietly became society-brittle.

    This test fits the real artifact via `train_final` on the real Gurgaon
    dataset and directly measures the gap between predictions with society
    supplied and predictions with society withheld (`_without_society`), on
    the same rows the model was fit on. With masking intact this gap is
    ~2.5pp; drop masking from just the final fit (leave `evaluate` alone) and
    it balloons to ~16pp -- the threshold below is set decisively between
    the two, verified by actually performing that mutation."""
    df = pd.read_csv("data/gurgaon/processed/listings_clean.csv")
    fitted = train_final(df)
    X_given = build_features(df, fitted.encoders)
    X_without = _without_society(X_given)
    pred_given = predict_price(fitted, X_given)
    pred_without = predict_price(fitted, X_without)
    actual = df["price"].to_numpy(dtype=float)
    mape_given = float(np.mean(np.abs(pred_given - actual) / actual) * 100)
    mape_without = float(np.mean(np.abs(pred_without - actual) / actual) * 100)
    gap = mape_without - mape_given
    assert gap >= -1e-9, "withholding real information should not IMPROVE the estimate"
    assert gap <= 8.0, (
        f"train_final's shipped fit shows a {gap:.2f}pp gap between society-given "
        f"and society-withheld predictions -- expected ~2.5pp. This is the "
        f"signature of society masking having been dropped from the FINAL fit "
        f"(model.py train_final's own `_mask_society` call), even if "
        f"evaluate()'s fold loop is still masked correctly.")


# ── no target leakage in the cross-validation loop ───────────────────────────
# The project's headline claim is that every fold-local encoding is
# re-learned inside every training fold. These tests fail if that ever stops
# being true.

@pytest.fixture()
def leak_prone_frame() -> pd.DataFrame:
    """A frame engineered so that leakage is measurable.

    One listing per sector, with prices spanning two orders of magnitude.
    Encoding the sector on the FULL frame therefore hands every row its own
    price-per-sqft, from which its price is exactly recoverable. Encoding
    fold-locally leaves each held-out sector unseen, so `sector_ppsf` falls
    back to the global median and carries no information about that row. The
    gap between the two scores is the leakage this project claims not to have.
    """
    n = 120
    rng = np.random.default_rng(11)
    area = rng.uniform(800.0, 1200.0, n).round()
    ppsf = np.exp(rng.uniform(np.log(2000.0), np.log(80000.0), n)).round()
    return pd.DataFrame({
        "sector": [f"sector {i}" for i in range(n)],
        "society": [f"society {i}" for i in range(n)],
        "property_type": "flat",
        "price": area * ppsf / 1e7,
        "price_per_sqft": ppsf,
        "area": area,
        "bedrooms": 3,
        "bathrooms": 2,
        "balcony": "2",
        "furnishing_type": "semi-furnished",
        "luxury_score": 50,
        "age_possession": "New Property",
    })


def _leaky_oof_r2(df: pd.DataFrame, n_splits: int = 5) -> float:
    """The wrong way, kept only as the thing `evaluate` must differ from.

    Identical to `evaluate`'s fold loop except every encoding is learned once
    from the whole frame -- including the held-out rows' own prices --
    instead of from each training fold. Never call this outside this test.
    """
    df = df.reset_index(drop=True)
    enc_full = fit_encoders(df)                      # <- the leak
    oof = np.zeros(len(df))
    for tr_idx, te_idx in KFold(n_splits, shuffle=True, random_state=7).split(df):
        tr, te = df.iloc[tr_idx], df.iloc[te_idx]
        est = GradientBoostingRegressor(**DEFAULT_PARAMS)
        est.fit(build_features(tr, enc_full), target(tr))
        oof[te_idx] = np.exp(est.predict(build_features(te, enc_full)))
    actual = df["price"].to_numpy(dtype=float)
    err = oof - actual
    return float(1 - np.sum(err ** 2) / np.sum((actual - actual.mean()) ** 2))


def test_evaluate_does_not_leak_the_target_through_encoders(leak_prone_frame):
    """Behavioural: evaluate() must not score like the leaky variant."""
    honest = evaluate(leak_prone_frame)["model"]["r2"]
    leaky = _leaky_oof_r2(leak_prone_frame)
    # the leaky loop nearly memorises the target on this frame
    assert leaky > 0.90, f"leaky reference should be near-perfect, got {leaky:.3f}"
    # an honest loop cannot come close, because every held-out sector/society
    # is unseen
    assert honest < leaky - 0.30, (
        f"evaluate() scored R2={honest:.3f} against a leaky R2={leaky:.3f} — "
        f"the fold loop looks like it is encoding on the full frame")


def _spy_on_fit_encoders(monkeypatch) -> list[int]:
    """Record the row count of every frame `model.fit_encoders` is called with."""
    import homecast.model as model_mod
    seen: list[int] = []
    real = model_mod.fit_encoders

    def spy(frame):
        seen.append(len(frame))
        return real(frame)

    monkeypatch.setattr(model_mod, "fit_encoders", spy)
    return seen


def test_fold_loop_encodes_on_training_rows_only(clean_fixture, monkeypatch):
    """Structural: every encoding inside evaluate() sees a training fold, never
    the whole frame."""
    seen = _spy_on_fit_encoders(monkeypatch)
    n = len(clean_fixture)
    evaluate(clean_fixture, n_splits=5)
    assert seen, "fit_encoders was never called — the spy is not wired up"
    assert n not in seen, (
        f"fit_encoders was handed all {n} rows: the fold loop is leaking")
    # KFold(5) training folds hold 4/5 of the rows, +-1 for the remainder
    assert all(abs(k - 0.8 * n) <= 1 for k in seen), seen


@pytest.mark.parametrize("k", [2, 3, 5])
def test_evaluate_runs_the_requested_number_of_folds(clean_fixture, monkeypatch, k):
    """`n_splits` is reported in the metrics; it must also be obeyed."""
    seen = _spy_on_fit_encoders(monkeypatch)
    out = evaluate(clean_fixture, n_splits=k)
    assert out["n_splits"] == k
    assert len(seen) == k
    n = len(clean_fixture)
    expected = n - n / k
    assert all(abs(rows - expected) <= 1 for rows in seen), seen
