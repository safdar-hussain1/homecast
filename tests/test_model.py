import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from homecast.features import build_features, fit_encoders, target
from homecast.model import DEFAULT_PARAMS, evaluate, predict_price, train_final


def test_evaluate_reports_model_and_baselines(clean_fixture):
    r = evaluate(clean_fixture, n_splits=3)
    for k in ("model", "baseline_sector", "baseline_global"):
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
