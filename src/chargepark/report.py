"""Summary tables and plots."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .io import PriceReport, SessionReport
from .montecarlo import MCResult

# (label, column, format, scale)
METRICS = [
    ("Energy delivered (MWh)", "energy_mwh", "{:,.0f}", 1),
    ("Energy without cap (MWh)", "energy_unconstrained_mwh", "{:,.0f}", 1),
    ("Energy lost to cap (MWh)", "energy_lost_mwh", "{:,.1f}", 1),
    ("Energy lost to cap (%)", "energy_lost_pct", "{:.2f}", 1),
    ("Sessions curtailed (%)", "sessions_curtailed_pct", "{:.1f}", 1),
    ("Cap binding (h/yr)", "cap_binding_hours", "{:,.0f}", 1),
    ("Peak 15-min load (kW)", "peak_15min_kw", "{:,.0f}", 1),
    ("Peak 15-min load, no cap (kW)", "peak_15min_unconstrained_kw", "{:,.0f}", 1),
    ("Target SOC reached (%)", "soc_target_reached_pct", "{:.1f}", 1),
    ("Mean energy per session (kWh)", "mean_session_kwh", "{:.1f}", 1),
    ("Cost DA (kEUR)", "cost_da_eur", "{:,.1f}", 1e-3),
    ("Cost ID (kEUR)", "cost_id_eur", "{:,.1f}", 1e-3),
    ("Cost PPA (kEUR)", "cost_ppa_eur", "{:,.1f}", 1e-3),
    ("Total energy cost (kEUR)", "cost_eur", "{:,.1f}", 1e-3),
    ("Avg energy cost (EUR/MWh)", "avg_cost_eur_mwh", "{:.1f}", 1),
]


def summary_table(runs: pd.DataFrame) -> pd.DataFrame:
    """P10 / P50 / P90 over Monte Carlo runs, in display units."""
    rows = []
    for label, col, _, scale in METRICS:
        p10, p50, p90 = np.percentile(runs[col], [10, 50, 90]) * scale
        rows.append({"metric": label, "P10": p10, "P50": p50, "P90": p90})
    return pd.DataFrame(rows).set_index("metric")


def vehicle_info(rep, cfg: dict) -> str:
    """One line describing the vehicle catalog behind a run."""
    r = cfg["representatives"]
    src = sorted(rep.curves.meta["source"].unique())
    wt = f"weights {Path(r['weights_csv']).name}" if r["weights_csv"] else "equal weight per distinct curve"
    return (f"Vehicles: k = {rep.k_label} ({rep.k} representatives) from {len(rep.curves)} curves "
            f"({', '.join(src)}; {wt})")


def format_summary(res: MCResult, cfg: dict, srep: SessionReport, prep: PriceReport,
                   runtime_s: float, vehicles: str = "") -> str:
    site, proc = cfg["site"], cfg["procurement"]
    tab = summary_table(res.runs)
    w = 70
    price_src = prep.source()
    lines = [
        "=" * w,
        f"Sessions: {srep.n_kept:,} of {srep.n_read:,} kept | year {srep.year} | "
        f"runs: {len(res.runs)} | runtime {runtime_s:.1f} s",
        f"Grid cap {site['grid_cap_kw']:.0f} kW | charger {site['charger_kw']:.0f} kW | "
        f"LM {site['lm_mode']} | procurement {proc['mode']} (PPA {proc['share_ppa']:.0%} "
        f"at {proc['ppa_price_eur_mwh']:.0f} EUR/MWh)",
        f"Prices: {price_src}",
        *([vehicles] if vehicles else []),
        "-" * w,
        f"{'Metric':<34}{'P10':>12}{'P50':>12}{'P90':>12}",
    ]
    for label, _, fmt, _ in METRICS:
        r = tab.loc[label]
        lines.append(f"{label:<34}" + "".join(f"{fmt.format(r[p]):>12}" for p in ("P10", "P50", "P90")))
    lines += [
        "=" * w,
        "Energy procurement only (no grid fees, levies, taxes, VAT).",
        "Arrival SOC distribution and PPA price are PLACEHOLDER assumptions.",
    ]
    if not cfg["representatives"]["weights_csv"]:
        lines.append("Vehicle mix: equal weight per distinct Fastned curve, NOT market shares.")
    if not cfg["vehicles"]["temperature_derating"]["enabled"]:
        lines.append("Curves reflect about 30 C battery conditions (no temperature derating).")
    if proc["mode"] == "da_only":
        lines.append("No ID prices: non-PPA energy is priced at DA in the quarter hour it is used "
                     "(perfect forecast); balancing cost is not included.")
    if proc["mode"] == "forecast":
        lines.append("DA forecast is the in-sample average profile, so ID volumes understate real forecast error.")
    if prep.synthetic:
        lines.append("Prices are SYNTHETIC: the cost figures are NOT an estimate of real procurement cost.")
    return "\n".join(lines)


def plot_overview(res: MCResult, sessions: pd.DataFrame, cfg: dict, path: str | Path,
                  synthetic_prices: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    c = cfg["output"]["colors"]
    cap = cfg["site"]["grid_cap_kw"]
    f = res.first
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    series = [("uncapped", c["purple"], "Without load management"),
              ("capped", c["green"], "With load management")]

    # (a) busiest week, 1-min load
    load_u = f.uncapped.load_kw
    daily = pd.Series(load_u).groupby(np.arange(len(load_u)) // 1440).sum()
    d0 = max(int(daily.rolling(7).sum().idxmax()) - 6, 0)
    sl = slice(d0 * 1440, (d0 + 7) * 1440)
    t = f.index_15[0] + pd.to_timedelta(np.arange(sl.start, sl.stop), unit="min")
    a = ax[0, 0]
    for key, col, lab in series:
        a.plot(t, getattr(f, key).load_kw[sl], color=col, lw=0.8, label=lab,
               alpha=0.6 if key == "uncapped" else 1.0)
    a.axhline(cap, color=c["pink"], lw=1.5, ls="--", label=f"Grid cap {cap:.0f} kW")
    a.set_title("Busiest week, 1-min site load (run 1)", loc="left", fontweight="bold")
    a.set_ylabel("kW")
    a.legend(loc="upper right", frameon=False, fontsize=8)
    a.tick_params(axis="x", rotation=30)

    # (b) load duration curve, 15-min averages
    a = ax[0, 1]
    for key, col, lab in series:
        q = np.sort(getattr(f, key).load_kw.reshape(-1, 15).mean(axis=1))[::-1]
        a.plot(np.arange(len(q)) / 4, q, color=col, lw=2, label=lab)
    a.axhline(cap, color=c["pink"], lw=1.5, ls="--", label=f"Grid cap {cap:.0f} kW")
    a.set_xlim(0, 3000)
    a.set_title("Load duration curve, 15-min averages (top 3,000 h)", loc="left", fontweight="bold")
    a.set_xlabel("Hours per year")
    a.set_ylabel("kW")
    a.legend(frameon=False, fontsize=8)

    # (c) energy per session vs plug-in duration
    a = ax[1, 0]
    d = sessions["duration_min"].to_numpy()
    for key, col, lab in series:
        a.scatter(d, getattr(f, key).energy_kwh, s=8, color=col, alpha=0.3, label=lab, linewidths=0)
    a.set_xlim(0, np.percentile(d, 99.5))
    a.set_title("Energy per session vs. plug-in duration (run 1)", loc="left", fontweight="bold")
    a.set_xlabel("Plug-in duration (min)")
    a.set_ylabel("Grid energy (kWh)")
    a.legend(frameon=False, fontsize=8, markerscale=2)

    # (d) Monte Carlo distribution of average cost
    a = ax[1, 1]
    v = res.runs["avg_cost_eur_mwh"]
    a.hist(v, bins=min(25, max(5, len(v) // 2)), color=c["blue"], alpha=0.75, edgecolor="white", linewidth=2)
    for p, ls in [(10, ":"), (50, "-"), (90, ":")]:
        x = np.percentile(v, p)
        a.axvline(x, color=c["navy"], ls=ls, lw=1.2)
        a.text(x, a.get_ylim()[1] * 0.95, f" P{p}\n {x:.1f}", fontsize=8, color=c["navy"], va="top")
    tag = " (SYNTHETIC prices)" if synthetic_prices else ""
    a.set_title(f"Average energy cost, {len(v)} Monte Carlo runs{tag}", loc="left", fontweight="bold")
    a.set_xlabel("EUR/MWh (energy procurement only)")
    a.set_ylabel("Runs")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
