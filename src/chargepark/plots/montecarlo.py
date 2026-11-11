"""06 Monte Carlo: spread of the KPIs over runs, convergence, profile band."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import hour_axis, percentile_lines
from .data import RunData
from .style import BLUE, GREEN, NAVY, PINK, figure, finish, tight, tint

KPIS = [
    ("energy_mwh", "Energy delivered (MWh)", 1),
    ("energy_lost_pct", "Energy lost to cap (%)", 1),
    ("cap_binding_hours", "Cap binding (h/yr)", 1),
    ("peak_15min_kw", "Peak 15-min load (kW)", 1),
    ("cost_eur", "Total energy cost (kEUR)", 1e-3),
    ("avg_cost_eur_mwh", "Average energy cost (EUR/MWh)", 1),
]


def kpi_distributions(d: RunData):
    mc = d.mc
    ref = d.meta["reference_run"]
    fig, axes = figure(2, 3, size=(12, 6.8))
    for ax, (col, lab, sc) in zip(axes.ravel(), KPIS):
        v = mc[col].to_numpy() * sc
        bins = min(25, max(5, len(v) // 2))
        if np.ptp(v) < 1e-9:
            bins = 1
        ax.hist(v, bins=bins, color=tint(BLUE, 0.35), edgecolor="white", linewidth=0.8, label="Runs")
        percentile_lines(ax, v, color=NAVY)
        rv = float(mc.loc[mc["run"] == ref, col].iloc[0] * sc)
        ax.axvline(rv, color=PINK, lw=1.6, label=f"Reference run {ref}")
        ax.set_xlabel(lab)
        ax.set_ylabel("Runs (count)")
        p10, p50, p90 = np.percentile(v, [10, 50, 90])
        ax.set_title(f"P50 {p50:,.4g} (P10 {p10:,.4g}, P90 {p90:,.4g})", fontsize=8.5, fontweight="normal")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncols=4)
    fig.suptitle(f"KPI distributions over {len(mc)} Monte Carlo runs", x=0.01, ha="left", fontweight="bold",
                 fontsize=10)
    finish(fig, d.notes(prices=True, assumptions=True))
    tight(fig, rect=(0, 0.06, 1, 0.94))
    return fig


def convergence(d: RunData):
    mc = d.mc.sort_values("run")
    n = np.arange(1, len(mc) + 1)
    fig, axes = figure(2, 1, size=(10, 6.5), sharex=True)
    for ax, (col, lab, sc), c in ((axes[0], KPIS[4], GREEN), (axes[1], KPIS[5], BLUE)):
        s = pd.Series(mc[col].to_numpy() * sc)
        ex = s.expanding()
        ax.fill_between(n, ex.quantile(0.1), ex.quantile(0.9), color=tint(c, 0.75), lw=0,
                        label="Running P10 to P90")
        ax.plot(n, ex.mean(), color=c, lw=1.8, label="Running mean")
        ax.plot(n, s, color="#999999", lw=0, marker="o", ms=2.5, label="Single run")
        ax.set_ylabel(lab)
        ax.legend(loc="upper right", ncols=3)
    axes[0].set_title(f"Convergence of the cost estimate over {len(mc)} runs")
    axes[1].set_xlabel("Number of Monte Carlo runs (count)")
    axes[1].set_xlim(1, max(len(mc), 2))
    finish(fig, d.notes(prices=True))
    return fig


def profile_band(d: RunData):
    prof = d.profiles
    fig, axes = figure(1, 2, size=(11.5, 4.8), sharey=True)
    if prof is None:
        for ax in axes:
            ax.text(0.5, 0.5, "mc_profiles_15min.parquet not found", ha="center", transform=ax.transAxes)
        return fig
    idx = prof.index
    slot = idx.hour * 4 + idx.minute // 15
    weekend = idx.day_of_week >= 5
    ref = f"run_{d.meta['reference_run']:03d}"
    x = np.arange(96) / 4 + 0.125
    for ax, mask, what in ((axes[0], ~weekend, "Mon to Fri"), (axes[1], weekend, "Sat and Sun")):
        avg = prof[mask].groupby(slot[mask]).mean()          # (96 slots, runs)
        p10, p50, p90 = (avg.quantile(q, axis=1) for q in (0.1, 0.5, 0.9))
        lo, hi = avg.min(axis=1), avg.max(axis=1)
        ax.fill_between(x, lo, hi, color=tint(GREEN, 0.85), lw=0, label="Min to max over runs")
        ax.fill_between(x, p10, p90, color=tint(GREEN, 0.55), lw=0, label="P10 to P90 over runs")
        ax.plot(x, p50, color=GREEN, lw=1.8, label="P50 over runs")
        if ref in avg:
            ax.plot(x, avg[ref], color=NAVY, lw=1.0, ls="--", label="Reference run")
        hour_axis(ax)
        ax.set_title(f"{what}", fontweight="normal")
    axes[0].set_ylabel("Grid load (kW, mean per 15 min)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper right", ncols=4)
    fig.suptitle(f"Average daily load profile across {prof.shape[1]} Monte Carlo runs", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    finish(fig, ["Each run samples vehicle types and arrival SOC anew; sessions (arrival, departure) are fixed."])
    tight(fig, rect=(0, 0.04, 1, 0.94))
    return fig
