"""Persist a Monte Carlo run to results/<run_id>/ for plotting and audit.

The plotting package (chargepark.plots) reads ONLY these files and never
re-runs the simulation. Layout (see CLAUDE.md):

  config_snapshot.yaml            effective config of the run
  run_meta.json                   run_id, inputs, year, reference run, flags
  mc_summary.csv                  one row per Monte Carlo run (KPIs)
  mc_profiles_15min.parquet       mean grid kW per 15 min (0.1 kW), one float32 column per run
  example_days.csv                rule, label, date, value, unit, reason
  catalog_curves.parquet          vehicle catalog: one row per representative
  curve_members.parquet           every eligible curve with its representative
  -- reference run (total cost closest to the median) only --
  load_1min.parquet               timestamp, load_kw, load_uncapped_kw, n_active_sessions, cap_binding
  qh.parquet                      15-min energy, prices, volumes and cost
  sessions.parquet                one row per session
  session_power_examples.parquet  per-minute power of every session on the example days
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .curves import SOC_COLS
from .examples import select_example_days
from .montecarlo import MCResult
from .procurement import procurement_breakdown, to_quarter_hours
from .simulate import SessionArrays, simulate
from .vehicles import Catalog

PARQUET = {"engine": "pyarrow", "compression": "zstd", "index": False}


def config_hash(cfg: dict, inputs: dict) -> str:
    blob = json.dumps({"config": cfg, "inputs": inputs}, sort_keys=True, default=str)
    return hashlib.sha1(blob.encode("utf-8")).hexdigest()[:8]


def new_run_dir(root: str | Path, cfg: dict, inputs: dict, now: dt.datetime | None = None) -> Path:
    """results/<YYYYmmdd-HHMMSS>-<config hash>/ (suffix _2, _3 on a clash)."""
    now = now or dt.datetime.now()
    base = Path(root) / f"{now:%Y%m%d-%H%M%S}-{config_hash(cfg, inputs)}"
    d, i = base, 2
    while d.exists():
        d, i = Path(f"{base}_{i}"), i + 1
    d.mkdir(parents=True)
    return d


def _curve_frames(catalog: Catalog, rep=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    if rep is not None:
        ids = rep.table["medoid_id"].astype(str).to_numpy()
    else:
        ids = np.array(catalog.names, dtype=object)
    cat = pd.DataFrame({"rep_index": np.arange(len(catalog)), "curve_id": ids, "name": catalog.names,
                        "battery_kwh": catalog.battery_kwh, "share": catalog.share})
    cat = pd.concat([cat, pd.DataFrame(catalog.power_kw, columns=SOC_COLS)], axis=1)
    if rep is None:
        mem = cat[["rep_index", "curve_id", "name", "battery_kwh"]].assign(weight=catalog.share, is_medoid=True)
        mem = pd.concat([mem, pd.DataFrame(catalog.power_kw, columns=SOC_COLS)], axis=1)
    else:
        m = rep.members
        names = (m["brand"] + " " + m["model"] + " " + m["version"].astype(str)).str.strip()
        mem = pd.DataFrame({"rep_index": m["cluster"].to_numpy() - 1, "curve_id": m["curve_id"].astype(str),
                            "name": names, "battery_kwh": m["battery_kwh"].to_numpy(float),
                            "weight": rep.weights,
                            "is_medoid": np.isin(np.arange(len(m)), rep.medoids)})
        mem = pd.concat([mem, pd.DataFrame(rep.curves.power_kw, columns=SOC_COLS)], axis=1)
    return cat, mem


def write_results(run_dir: str | Path, res: MCResult, sessions: pd.DataFrame, year: int,
                  prices: pd.DataFrame, cfg: dict, catalog: Catalog, rep=None,
                  meta: dict | None = None) -> Path:
    """Write every result file. Re-simulates only the reference run once more,
    with per-minute tracing of the sessions on the example days."""
    out = Path(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    ref = res.reference
    idx15 = ref.index_15
    t0 = idx15[0]
    sa = SessionArrays.from_frame(sessions, year)
    n_min = sa.n_minutes
    minutes = pd.date_range(t0, periods=n_min, freq="min")
    cap, free = ref.capped, ref.uncapped

    (out / "config_snapshot.yaml").write_text(
        yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    res.runs.to_csv(out / "mc_summary.csv", index=False)
    prof = pd.DataFrame(np.round(res.profiles_kw.T, 1), columns=[f"run_{r:03d}" for r in res.runs["run"]])
    prof.insert(0, "timestamp", idx15)
    prof.to_parquet(out / "mc_profiles_15min.parquet", **PARQUET)

    load = pd.DataFrame({
        "load_kw": cap.load_kw.astype(np.float32),
        "load_uncapped_kw": free.load_kw.astype(np.float32),
        "n_active_sessions": cap.n_active.astype(np.int16),
        "cap_binding": cap.binding,
    }, index=minutes)
    qh = procurement_breakdown(to_quarter_hours(cap.load_kw, idx15), prices, cfg)
    ref_cost = res.runs.loc[res.runs["run"] == ref.run, "cost_eur"].iloc[0]
    if not np.isclose(qh["cost_eur"].sum(), ref_cost, rtol=1e-9, atol=1e-6):
        raise RuntimeError("reference run re-simulation does not reproduce its Monte Carlo cost")

    ex = select_example_days(load, qh)
    ex.to_csv(out / "example_days.csv", index=False)
    load.rename_axis("timestamp").reset_index().to_parquet(out / "load_1min.parquet", **PARQUET)
    qh.rename_axis("timestamp").reset_index().to_parquet(out / "qh.parquet", **PARQUET)

    # sessions
    tol = cfg["montecarlo"]["curtailed_tol_kwh"]
    target = cfg["vehicles"]["target_soc"]
    vt = ref.fleet.vtype
    cat_df, mem_df = _curve_frames(catalog, rep)
    sess = pd.DataFrame({
        "session_id": np.arange(len(sa), dtype=np.int64),
        "start": t0 + pd.to_timedelta(sa.start_min, unit="min"),
        "end": t0 + pd.to_timedelta(sa.end_min, unit="min"),
        "duration_min": sa.end_min - sa.start_min,
        "charger_kw": sa.charger_kw,
        "rep_index": vt,
        "representative_curve_id": cat_df["curve_id"].to_numpy()[vt],
        "battery_kwh": ref.fleet.capacity_kwh,
        "soc_arrival": ref.fleet.soc0,
        "soc_end": cap.soc_end,
        "energy_kwh": cap.energy_kwh,
        "energy_uncapped_kwh": free.energy_kwh,
        "curtailed": cap.energy_kwh < free.energy_kwh - tol,
        "target_reached": cap.soc_end >= target - 1e-6,
    })
    if "charger_id" in sessions:
        sess.insert(3, "charger_id", sessions["charger_id"].to_numpy())
    sess.to_parquet(out / "sessions.parquet", **PARQUET)
    cat_df.to_parquet(out / "catalog_curves.parquet", **PARQUET)
    mem_df.to_parquet(out / "curve_members.parquet", **PARQUET)

    # per-minute detail of every session plugged in on an example day
    days = pd.to_datetime(sorted(ex["date"].unique()))
    day_min = ((days - t0) / pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
    on_day = np.zeros(len(sa), bool)
    for d0 in day_min:
        on_day |= (sa.start_min < d0 + 1440) & (sa.end_min > d0)
    traced = simulate(sa, ref.fleet, catalog.power_kw, cfg, trace=np.flatnonzero(on_day))
    if not np.allclose(traced.load_kw, cap.load_kw):
        raise RuntimeError("traced re-simulation differs from the reference run")
    tr = traced.trace
    tr.insert(1, "timestamp", t0 + pd.to_timedelta(tr.pop("minute"), unit="min"))
    for c in ("power_kw", "requested_kw", "soc"):
        tr[c] = tr[c].astype(np.float32)
    tr.to_parquet(out / "session_power_examples.parquet", **PARQUET)

    m = {"run_id": out.name, "created": dt.datetime.now().isoformat(timespec="seconds"),
         "year": int(year), "n_sessions": int(len(sa)), "runs": int(len(res.runs)),
         "reference_run": int(ref.run), "has_id_prices": bool("id" in prices),
         "grid_cap_kw": float(cfg["site"]["grid_cap_kw"]),
         "local_timezone": cfg["site"]["local_timezone"], **(meta or {})}
    (out / "run_meta.json").write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
    return out
