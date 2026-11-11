"""Monte Carlo loop: sample fleets, simulate with and without the cap, price."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .io import year_index_15
from .procurement import price_energy, to_quarter_hours
from .simulate import SessionArrays, SimResult, simulate
from .vehicles import Catalog, Fleet, sample_fleet


@dataclass
class ReferenceRun:
    """Full detail of the reference run: the run whose total cost is closest
    to the median over runs (ties: lowest run number)."""
    run: int
    capped: SimResult
    uncapped: SimResult
    fleet: Fleet
    index_15: pd.DatetimeIndex


@dataclass
class MCResult:
    runs: pd.DataFrame         # one row per run
    reference: ReferenceRun
    fleets: list[Fleet]        # sampled fleet per run
    profiles_kw: np.ndarray    # (runs, quarter hours) float32, mean grid kW per 15 min (capped)

    @property
    def first(self) -> ReferenceRun:          # backwards compatible name
        return self.reference


def reference_run(runs: pd.DataFrame) -> int:
    """Run whose total cost is closest to the median (lowest run on ties)."""
    c = runs["cost_eur"].to_numpy()
    dev = np.abs(c - np.median(c))
    return int(runs["run"].to_numpy()[np.flatnonzero(dev == dev.min())[0]])


def run_montecarlo(sessions: pd.DataFrame, year: int, prices: pd.DataFrame, cfg: dict,
                   catalog: Catalog, runs: int | None = None) -> MCResult:
    runs = int(runs or cfg["montecarlo"]["runs"])
    sa = SessionArrays.from_frame(sessions, year)
    idx15 = year_index_15(year)
    if not prices.index.equals(idx15):
        raise ValueError("prices must be on the 15-min grid of the modeled year")
    rng = np.random.default_rng(cfg["montecarlo"]["seed"])
    target = cfg["vehicles"]["target_soc"]
    tol = cfg["montecarlo"]["curtailed_tol_kwh"]

    rows, fleets = [], []
    profiles = np.empty((runs, len(idx15)), np.float32)
    for r in range(runs):
        fleet = sample_fleet(len(sa), catalog, cfg, rng)
        fleets.append(fleet)
        cap = simulate(sa, fleet, catalog.power_kw, cfg)
        free = simulate(sa, fleet, catalog.power_kw, cfg, grid_cap_kw=np.inf)
        e_qh = to_quarter_hours(cap.load_kw, idx15)
        profiles[r] = e_qh.to_numpy() * 4
        cost = price_energy(e_qh, prices, cfg)
        e_mwh = cap.energy_kwh.sum() / 1000
        e_free_mwh = free.energy_kwh.sum() / 1000
        rows.append({
            "run": r,
            "energy_mwh": e_mwh,
            "energy_unconstrained_mwh": e_free_mwh,
            "energy_lost_mwh": e_free_mwh - e_mwh,
            "energy_lost_pct": 100 * (1 - e_mwh / e_free_mwh) if e_free_mwh > 0 else 0.0,
            "sessions_curtailed_pct": 100 * np.mean(cap.energy_kwh < free.energy_kwh - tol),
            "cap_binding_hours": cap.binding_minutes / 60,
            "peak_15min_kw": float(e_qh.max() * 4),
            "peak_15min_unconstrained_kw": float(to_quarter_hours(free.load_kw, idx15).max() * 4),
            "soc_target_reached_pct": 100 * np.mean(cap.soc_end >= target - 1e-6),
            "mean_session_kwh": float(cap.energy_kwh.mean()),
            "avg_cost_eur_mwh": cost["cost_eur"] / e_mwh if e_mwh > 0 else np.nan,
            **cost,
        })
    df = pd.DataFrame(rows)
    ref = reference_run(df)
    f = fleets[ref]
    ref_run = ReferenceRun(ref, simulate(sa, f, catalog.power_kw, cfg),
                           simulate(sa, f, catalog.power_kw, cfg, grid_cap_kw=np.inf), f, idx15)
    return MCResult(df, ref_run, fleets, profiles)


def compare_k(sessions: pd.DataFrame, year: int, prices: pd.DataFrame, cfg: dict, ks,
              runs: int | None = None) -> pd.DataFrame:
    """Run the Monte Carlo for several k (same sessions, prices and seed) and
    compare P50 results with the k = "all" reference (added if missing)."""
    from .curves import load_curves
    from .representatives import build_representatives, overall_errors
    from .vehicles import Catalog

    ks = list(ks)
    if "all" not in ks:
        ks = ["all"] + ks
    cs = load_curves(cfg)
    rows = []
    for k in ks:
        rep = build_representatives(cs, cfg, k)
        r = run_montecarlo(sessions, year, prices, cfg, Catalog.from_representatives(rep), runs).runs
        p = lambda c, q=50: float(np.percentile(r[c], q))
        err = overall_errors(rep)
        rows.append({
            "k": rep.k_label, "n_types": rep.k,
            "curve_mae_kw": err["mae_kw"], "t10_80_mad_min": err["t10_80_mad_min"],
            "energy_mwh": p("energy_mwh"), "energy_mwh_p10": p("energy_mwh", 10),
            "energy_mwh_p90": p("energy_mwh", 90),
            "energy_lost_mwh": p("energy_lost_mwh"), "energy_lost_pct": p("energy_lost_pct"),
            "cap_binding_hours": p("cap_binding_hours"),
            "eur_mwh": p("avg_cost_eur_mwh"), "eur_mwh_p10": p("avg_cost_eur_mwh", 10),
            "eur_mwh_p90": p("avg_cost_eur_mwh", 90),
        })
    df = pd.DataFrame(rows)
    ref = df[df["k"] == "all"].iloc[0]
    df["energy_dev_pct"] = 100 * (df["energy_mwh"] / ref["energy_mwh"] - 1)
    df["energy_lost_dev_mwh"] = df["energy_lost_mwh"] - ref["energy_lost_mwh"]
    df["eur_mwh_dev"] = df["eur_mwh"] - ref["eur_mwh"]
    return df
