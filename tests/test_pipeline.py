"""End-to-end smoke tests on a small synthetic session file."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from chargepark.cli import main
from chargepark.io import load_sessions, synthetic_prices
from chargepark.montecarlo import run_montecarlo
from chargepark.report import summary_table
from chargepark.vehicles import Catalog

FASTNED = Path(__file__).resolve().parents[1] / "data" / "ev_curves" / "fastned" / "out"
needs_fastned = pytest.mark.skipif(not (FASTNED / "curves_wide.csv").exists(),
                                   reason="Fastned databank not extracted")
CAT = Catalog.flat([60, 120, 200], [50, 75, 95])


@pytest.fixture
def small_sessions(tmp_path):
    rng = np.random.default_rng(0)
    days = pd.date_range("2025-01-01", "2025-01-14", freq="D")
    rows = []
    for d in days:
        for m in np.sort(rng.integers(6 * 60, 22 * 60, 40)):
            dur = int(rng.integers(10, 60))
            rows.append((d.strftime("%Y-%m-%d"), f"{m // 60:02d}:{m % 60:02d}",
                         f"{(m + dur) % 1440 // 60:02d}:{(m + dur) % 60:02d}"))
    rows.append(("2025-01-05", "23:40", "00:25"))             # crosses midnight
    p = tmp_path / "sessions.csv"
    pd.DataFrame(rows, columns=["date", "arrival_time", "end_time"]).to_csv(p, index=False)
    return p


@pytest.fixture
def tmp_config(tmp_path, cfg):
    """Config file whose outputs go to tmp_path."""
    def make(**changes):
        c = dict(cfg)
        c["representatives"] = {**cfg["representatives"], "output_dir": str(tmp_path / "processed"),
                                **changes.get("representatives", {})}
        c["prices"] = {**cfg["prices"], **changes.get("prices", {})}
        p = tmp_path / "cfg.yaml"
        p.write_text(yaml.safe_dump(c), encoding="utf-8")
        return p
    return make


def test_montecarlo_metrics_consistent(small_sessions, cfg):
    sessions, rep = load_sessions(small_sessions, cfg)
    prices, _ = synthetic_prices(rep.year, cfg)
    res = run_montecarlo(sessions, rep.year, prices, cfg, CAT, runs=4)
    r = res.runs
    assert len(r) == 4
    assert (r["energy_mwh"] <= r["energy_unconstrained_mwh"] + 1e-9).all()
    assert (r["peak_15min_kw"] <= cfg["site"]["grid_cap_kw"] + 1e-6).all()
    assert np.allclose(r["cost_eur"], r["cost_da_eur"] + r["cost_id_eur"] + r["cost_ppa_eur"])
    assert np.allclose(r["vol_da_mwh"] + r["vol_id_net_mwh"] + r["vol_ppa_mwh"], r["energy_mwh"])
    tab = summary_table(r)
    assert (tab["P10"] <= tab["P50"]).all() and (tab["P50"] <= tab["P90"]).all()


def test_montecarlo_reproducible(small_sessions, cfg):
    sessions, rep = load_sessions(small_sessions, cfg)
    prices, _ = synthetic_prices(rep.year, cfg)
    a = run_montecarlo(sessions, rep.year, prices, cfg, CAT, runs=2).runs
    b = run_montecarlo(sessions, rep.year, prices, cfg, CAT, runs=2).runs
    pd.testing.assert_frame_equal(a, b)


def test_cli_requires_price_source(small_sessions, tmp_path, tmp_config):
    c = tmp_config(prices={"da": {"path": None, "variable": "x", "timestamp": "t", "timestamps_tz": "local"}})
    assert main(["run", "--config", str(c), "--sessions", str(small_sessions),
                 "--out", str(tmp_path / "o")]) == 2


@needs_fastned
def test_cli_run_end_to_end(small_sessions, tmp_path, tmp_config, capsys):
    out = tmp_path / "o"
    rc = main(["--config", str(tmp_config()), "--sessions", str(small_sessions),   # no subcommand = run
               "--synthetic-prices", "--runs", "3", "--k", "4", "--out", str(out)])
    assert rc == 0
    (run,) = list(out.iterdir())                                                   # results/<run_id>/
    assert len(run.name.split("-")[-1]) == 8                                       # config hash
    for f in ("summary.txt", "summary.csv", "mc_summary.csv", "overview.png", "vehicle_catalog.csv",
              "config_snapshot.yaml", "run_meta.json", "example_days.csv", "load_1min.parquet",
              "qh.parquet", "sessions.parquet", "session_power_examples.parquet",
              "mc_profiles_15min.parquet", "catalog_curves.parquet", "curve_members.parquet"):
        assert (run / f).exists(), f
    text = capsys.readouterr().out
    assert "SYNTHETIC" in text and "P50" in text and "k = 4" in text
    assert "—" not in (run / "summary.txt").read_text(encoding="utf-8")   # no em dashes


@needs_fastned
def test_cli_curves_and_elbow(tmp_path, tmp_config):
    c = tmp_config(representatives={"elbow_k_max": 4})
    assert main(["curves", "--config", str(c), "--k", "3", "--elbow"]) == 0
    d = tmp_path / "processed"
    for f in ("representatives_k3.csv", "representatives_k3_clusters.png", "elbow.csv", "elbow.png"):
        assert (d / f).exists()
    assert list(pd.read_csv(d / "elbow.csv")["k"]) == [2, 3, 4]


@needs_fastned
def test_compare_k_reports_deviation_from_all(small_sessions, tmp_path, tmp_config):
    out = tmp_path / "o"
    assert main(["compare-k", "--config", str(tmp_config()), "--sessions", str(small_sessions),
                 "--synthetic-prices", "--runs", "2", "--ks", "3", "--out", str(out)]) == 0
    df = pd.read_csv(out / "compare_k.csv")
    assert list(df["k"].astype(str)) == ["all", "3"]
    assert df.loc[0, "energy_dev_pct"] == 0.0
