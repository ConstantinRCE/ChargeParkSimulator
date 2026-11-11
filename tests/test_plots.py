"""Results persistence, example-day selection and the plot package."""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from chargepark.examples import RULES, select_example_days
from chargepark.io import load_sessions, synthetic_prices
from chargepark.montecarlo import run_montecarlo
from chargepark.plots.catalog import SECTIONS, build_catalog, section_key
from chargepark.plots.cli import plot_dir_for, run_plots
from chargepark.plots.data import RunData, day_matrix, dst_info
from chargepark.plots.style import save
from chargepark.results import new_run_dir, write_results
from chargepark.vehicles import Catalog

ROOT = Path(__file__).resolve().parents[1]
DUMMY = ROOT / "data" / "sessions_dummy.csv"
FASTNED = ROOT / "data" / "ev_curves" / "fastned" / "out" / "curves_wide.csv"
DA_2025 = ROOT / "data" / "prices" / "DA_2025_15min_Germany_SMARD.nc"
CAT = Catalog.flat([60, 120, 200], [50, 75, 95])


def _sessions_csv(path: Path) -> Path:
    """Busy days around both DST changes plus a few quiet days."""
    rng = np.random.default_rng(1)
    days = list(pd.date_range("2025-03-28", "2025-04-02")) + list(pd.date_range("2025-10-24", "2025-10-28")) \
        + [pd.Timestamp("2025-01-15"), pd.Timestamp("2025-07-05"), pd.Timestamp("2025-07-06")]
    rows = []
    for d in days:
        for m in np.sort(rng.integers(6 * 60, 22 * 60, 30)):
            dur = int(rng.integers(15, 70))
            rows.append((d.strftime("%Y-%m-%d"), f"{m // 60:02d}:{m % 60:02d}",
                         f"{(m + dur) % 1440 // 60:02d}:{(m + dur) % 60:02d}"))
    rows.append(("2025-03-29", "23:30", "00:40"))           # crosses midnight
    pd.DataFrame(rows, columns=["date", "arrival_time", "end_time"]).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def run_dir(tmp_path_factory):
    from chargepark.config import load_config
    cfg = load_config()
    cfg["site"]["grid_cap_kw"] = 200.0          # forces curtailment
    tmp = tmp_path_factory.mktemp("res")
    sessions, rep = load_sessions(_sessions_csv(tmp / "s.csv"), cfg)
    prices, prep = synthetic_prices(rep.year, cfg)
    res = run_montecarlo(sessions, rep.year, prices, cfg, CAT, runs=4)
    inputs = {"sessions": "s.csv", "synthetic_prices": True}
    out = new_run_dir(tmp / "results", cfg, inputs)
    write_results(out, res, sessions, rep.year, prices, cfg, CAT, meta={**inputs, "price_source": prep.source()})
    return out


@pytest.fixture(scope="module")
def data(run_dir):
    return RunData(run_dir)


# --- results files ---------------------------------------------------------------
def test_results_files_and_columns(data):
    l, q, s, tr = data.load, data.qh, data.sessions, data.trace
    assert len(l) == 365 * 1440 and len(q) == 365 * 96
    assert {"load_kw", "load_uncapped_kw", "n_active_sessions", "cap_binding"} <= set(l.columns)
    assert {"energy_kwh", "da_price", "id_price", "ppa_volume", "da_volume", "id_volume", "cost_eur"} <= set(q.columns)
    assert {"session_id", "start", "end", "representative_curve_id", "soc_arrival", "soc_end", "energy_kwh",
            "energy_uncapped_kwh", "curtailed", "target_reached"} <= set(s.columns)
    assert {"session_id", "timestamp", "power_kw", "requested_kw", "soc"} <= set(tr.columns)
    # reference run is the run whose cost is closest to the median, and its files reproduce it
    mc = data.mc
    ref = data.meta["reference_run"]
    dev = (mc["cost_eur"] - mc["cost_eur"].median()).abs()
    assert dev[mc["run"] == ref].iloc[0] == dev.min()
    assert np.isclose(q["cost_eur"].sum(), mc.loc[mc["run"] == ref, "cost_eur"].iloc[0])
    assert np.isclose(s["energy_kwh"].sum(), q["energy_kwh"].sum(), rtol=1e-6)
    assert np.isclose(l["load_kw"].sum() / 60, s["energy_kwh"].sum(), rtol=1e-5)
    assert s["curtailed"].any()
    assert (l["load_kw"] <= data.cap_kw + 1e-3).all()
    # every example day has its sessions traced, and the trace matches the load
    for day in data.example_dates["date"]:
        t = tr[(tr["timestamp"] >= day) & (tr["timestamp"] < day + pd.Timedelta(days=1))]
        tot = t.groupby("timestamp")["power_kw"].sum().reindex(
            pd.date_range(day, periods=1440, freq="min"), fill_value=0.0)
        assert np.allclose(tot.to_numpy(), l.loc[tot.index, "load_kw"].to_numpy(), atol=1e-2)
    assert (tr["power_kw"] <= tr["requested_kw"] + 1e-4).all()


def test_run_id_has_timestamp_and_config_hash(run_dir):
    stamp, h = run_dir.name.rsplit("-", 1)
    assert len(h) == 8 and len(stamp) == len("20250101-120000")


# --- example days ----------------------------------------------------------------
def test_example_days_deterministic(data):
    a = select_example_days(data.load, data.qh)
    b = select_example_days(data.load.sample(frac=1, random_state=3).sort_index(), data.qh)
    pd.testing.assert_frame_equal(a, b)
    assert list(a["rule"]) == list(RULES)
    assert a["reason"].str.len().gt(0).all()
    saved = pd.read_csv(data.run_dir / "example_days.csv")
    assert list(saved["date"]) == list(a["date"])


def test_example_days_rules_and_ties():
    idx = pd.date_range("2025-01-01", periods=4 * 1440, freq="min")
    load = pd.DataFrame({"load_kw": 10.0, "load_uncapped_kw": 10.0, "cap_binding": False}, index=idx)
    load.loc["2025-01-03 12:00":"2025-01-03 12:09", ["load_uncapped_kw", "cap_binding"]] = [900.0, True]
    q15 = pd.date_range("2025-01-01", periods=4 * 96, freq="15min")
    qh = pd.DataFrame({"energy_kwh": 2.5, "da_price": 50.0, "cost_eur": 0.1}, index=q15)
    qh.loc["2025-01-02", "da_price"] = -1.0
    qh.loc["2025-01-04", "cost_eur"] = 1.0
    ex = select_example_days(load, qh).set_index("rule")
    assert ex.loc["peak_load", "date"] == "2025-01-03"
    assert ex.loc["most_cap_minutes", "date"] == "2025-01-03"
    assert ex.loc["most_negative_prices", "date"] == "2025-01-02"
    assert ex.loc["most_expensive", "date"] == "2025-01-04"
    assert ex.loc["median_energy", "date"] == "2025-01-01"         # all equal: earliest date wins
    assert ex.loc["typical_weekday", "date"] == "2025-01-01"


# --- DST ---------------------------------------------------------------------------
def test_dst_info_and_day_matrix():
    info = dst_info(2025, "Europe/Berlin")
    assert info.spring_day == pd.Timestamp("2025-03-30") and info.spring_hours == [2]
    assert info.autumn_day == pd.Timestamp("2025-10-26") and info.autumn_hours == [2]
    s = pd.Series(1.0, index=pd.date_range("2025-01-01", "2025-12-31 23:45", freq="15min"))
    m = day_matrix(s, 96, "mean", info, 2025)
    assert m.shape == (365, 96)
    assert m.loc["2025-03-30", 8:11].isna().all() and m.isna().sum().sum() == 4
    assert m.loc["2025-10-26"].notna().all()
    assert "does not exist" in info.note() and "—" not in info.note()


# --- plots ---------------------------------------------------------------------------
def test_every_plot_function_writes_pdf_and_png(data, tmp_path):
    specs = build_catalog(data)
    assert {s.section for s in specs} == set(SECTIONS)
    assert len({s.name for s in specs}) == len(specs)
    for spec in specs:
        assert spec.name.startswith(spec.section + "_")
        fig = spec.make(data)
        for ax in fig.axes:                       # rules: units on labeled axes, no em dashes
            for lab in (ax.get_xlabel(), ax.get_ylabel()):
                assert "—" not in lab
        for t in fig.findobj(plt.Text):
            assert "—" not in t.get_text(), spec.name
        paths = save(fig, spec.name, tmp_path)
        plt.close(fig)
        for p in paths:
            assert p.exists() and p.stat().st_size > 1000, p


def test_axes_with_two_series_have_legend(data):
    """Every axes with 2+ labeled series carries a legend (own or figure level)."""
    for spec in build_catalog(data):
        fig = spec.make(data)
        has_fig_legend = bool(fig.legends)
        for ax in fig.axes:
            labeled = [h for h in ax.get_legend_handles_labels()[1] if not h.startswith("_")]
            if len(labeled) >= 2:
                assert ax.get_legend() is not None or has_fig_legend, spec.name
        plt.close(fig)


def test_run_plots_cli_report_and_only(run_dir):
    res = run_plots(run_dir, workers=1, verbose=False)
    out = res["out_dir"]
    assert out == plot_dir_for(run_dir) and out.parent.name == "plots"
    from pypdf import PdfReader
    n_plots = len(build_catalog(RunData(run_dir)))
    assert len(PdfReader(out / "report.pdf").pages) == n_plots + 2           # title + example days
    sub = run_plots(run_dir, only=["load"], report=False, workers=1, verbose=False,
                    out_dir=run_dir.parent / "only")
    assert {p.name.split("_")[0] for p in sub["files"]} == {"03"}


def test_section_key():
    assert section_key("3") == section_key("03") == section_key("load") == section_key("03_load") == "03"
    with pytest.raises(ValueError):
        section_key("nope")


def test_html_optional(run_dir, tmp_path):
    pytest.importorskip("plotly")
    from chargepark.plots.html import write_html
    p = write_html(RunData(run_dir), tmp_path / "x.html")
    assert p.stat().st_size > 10_000


def test_plots_never_import_the_simulation():
    import ast
    for f in (ROOT / "src" / "chargepark" / "plots").glob("*.py"):
        tree = ast.parse(f.read_text(encoding="utf-8"))
        mods = {n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) and n.module}
        assert not mods & {"simulate", "montecarlo", "results"} or f.name == "montecarlo.py", f
        # plots/montecarlo.py is the 06 section; it must not import the simulation either
        src = f.read_text(encoding="utf-8")
        assert "..simulate" not in src and "..montecarlo" not in src and "..results" not in src


# --- full run on the dummy data ----------------------------------------------------
@pytest.mark.skipif(not (DUMMY.exists() and FASTNED.exists() and DA_2025.exists()),
                    reason="dummy sessions, Fastned curves or 2025 DA prices missing")
def test_full_plot_run_on_dummy_data_under_60s(tmp_path, cfg, capsys):
    from chargepark.cli import main
    assert main(["run", "--sessions", str(DUMMY), "--runs", "10", "--no-plot", "--out", str(tmp_path)]) == 0
    (run,) = [p for p in tmp_path.iterdir() if p.is_dir()]
    t = time.time()
    res = run_plots(run, verbose=False)
    sec = time.time() - t
    with capsys.disabled():
        print(f"\nfull plot run: {sec:.1f} s, {len(res['files'])} files; largest plot PDF "
              f"{res['largest_pdf'].name} {res['largest_pdf_kb']:.0f} kB")
    assert sec < 60
    assert res["largest_pdf_kb"] < 2000
