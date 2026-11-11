import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chargepark.curves import N_SOC, CurveSet, load_curves
from chargepark.representatives import (build_representatives, charge_minutes, distance_matrix,
                                        elbow, pam, pam_cost, write_outputs)

FASTNED = Path(__file__).resolve().parents[1] / "data" / "ev_curves" / "fastned" / "out"
needs_fastned = pytest.mark.skipif(not (FASTNED / "curves_wide.csv").exists(),
                                   reason="Fastned databank not extracted")


def toy_set(curves, battery, brands=None):
    n = len(curves)
    brands = brands or [f"B{i}" for i in range(n)]
    meta = pd.DataFrame({
        "curve_id": [f"t:{i}" for i in range(n)], "source": "fastned", "brand": brands,
        "model": [f"M{i}" for i in range(n)], "version": "", "category": "CAR",
        "battery_kwh": np.asarray(battery, float), "peak_kw": np.max(curves, axis=1),
        "charger_class": "", "n_versions": 1})
    return CurveSet(meta, np.asarray(curves, float), meta[["curve_id", "brand", "model", "version"]].copy())


# ---------------------------------------------------------------- PAM
def test_pam_finds_one_medoid_per_blob():
    rng = np.random.default_rng(0)
    centers = np.array([[0, 0], [10, 0], [0, 10]])
    X = np.vstack([c + rng.normal(0, 0.3, (12, 2)) for c in centers])
    med, _ = pam(distance_matrix(X), np.ones(len(X)), 3)
    assert sorted(m // 12 for m in med) == [0, 1, 2]


@pytest.mark.parametrize("k", [2, 3])
def test_pam_matches_brute_force_optimum(k):
    rng = np.random.default_rng(k)
    X = rng.normal(size=(10, 3))
    w = rng.uniform(0.5, 3, 10)
    D = distance_matrix(X)
    best = min(pam_cost(D, w, c) for c in itertools.combinations(range(10), k))
    _, cost = pam(D, w, k, seed=1, n_init=5)
    assert cost == pytest.approx(best)


def test_pam_deterministic_for_fixed_seed():
    X = np.random.default_rng(3).normal(size=(60, 4))
    D = distance_matrix(X)
    a = pam(D, np.ones(60), 6, seed=7, n_init=4)
    b = pam(D, np.ones(60), 6, seed=7, n_init=4)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1]


# ---------------------------------------------------------------- features, capping, errors
def test_charge_minutes_flat_curve():
    # 70 kWh, flat 70 kW: 10 -> 80 % is 49 kWh at 70 kW = 42 min
    assert charge_minutes(np.full(N_SOC, 70.0), 70.0)[0] == pytest.approx(42.0)
    assert charge_minutes(np.full(N_SOC, 70.0), 70.0, vehicle_eff=0.9)[0] == pytest.approx(42.0 / 0.9)


def test_curves_identical_after_capping_cluster_together(cfg):
    # two curves differing only above the 150 kW charger cap, same battery -> distance 0
    flat = lambda kw: np.full(N_SOC, kw, float)
    cs = toy_set([flat(200), flat(250), flat(50), flat(60)], [80, 80, 40, 40])
    cfg["representatives"].update(k=2)
    rep = build_representatives(cs, cfg)
    assert rep.labels[0] == rep.labels[1] and rep.labels[2] == rep.labels[3]
    c01 = rep.labels[0]
    assert rep.table.loc[c01, "mae_kw"] == 0.0                     # identical once capped


def test_k_all_returns_every_curve_with_unit_weights(cfg):
    rng = np.random.default_rng(0)
    cs = toy_set(rng.uniform(20, 200, (7, N_SOC)), rng.uniform(40, 100, 7))
    rep = build_representatives(cs, cfg, k="all")
    assert rep.k == 7 and set(rep.table["medoid_id"]) == set(cs.meta["curve_id"])
    assert rep.table["weight"].sum() == pytest.approx(1.0)
    assert (rep.table["cluster_size"] == 1).all() and (rep.table["mae_kw"] == 0).all()


def test_external_weights_hook(tmp_path, cfg):
    flat = lambda kw: np.full(N_SOC, kw, float)
    cs = toy_set([flat(50), flat(150)], [50, 80])
    p = tmp_path / "w.csv"
    pd.DataFrame({"brand": ["B0", "B1"], "model": ["M0", "M1"], "weight": [3.0, 1.0]}).to_csv(p, index=False)
    cfg["representatives"]["weights_csv"] = str(p)
    rep = build_representatives(cs, cfg, k="all")
    w = rep.table.set_index("medoid_id")["weight"]
    assert w["t:0"] == pytest.approx(0.75) and w["t:1"] == pytest.approx(0.25)


# ---------------------------------------------------------------- real databank
@needs_fastned
def test_real_k_all_and_weights(cfg):
    cs = load_curves(cfg)
    rep = build_representatives(cs, cfg, k="all")
    assert rep.k == len(cs)
    assert rep.table["weight"].sum() == pytest.approx(1.0)


@needs_fastned
def test_real_clustering_deterministic(cfg):
    cs = load_curves(cfg)
    a = build_representatives(cs, cfg, k=5)
    b = build_representatives(cs, cfg, k=5)
    assert list(a.table["medoid_id"]) == list(b.table["medoid_id"])
    assert a.table["weight"].sum() == pytest.approx(1.0)
    assert a.table["cluster_size"].sum() == len(cs)


@needs_fastned
def test_real_elbow_error_decreases(cfg):
    df = elbow(load_curves(cfg), cfg, ks=[2, 5, 10, 20])
    assert df["cost"].is_monotonic_decreasing


@needs_fastned
def test_write_outputs(tmp_path, cfg):
    cfg["representatives"]["output_dir"] = str(tmp_path)
    rep = build_representatives(load_curves(cfg), cfg, k=3)
    paths = write_outputs(rep, cfg)
    assert all(p.exists() for p in paths)
    t = pd.read_csv(tmp_path / "representatives_k3.csv")
    assert {"medoid_id", "brand", "model", "battery_kwh", "weight", "cluster_size"} <= set(t.columns)
