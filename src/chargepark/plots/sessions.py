"""04 Sessions: SOC, energy and target attainment of the reference run."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import hour_axis
from .data import RunData
from .style import BLUE, GREEN, NAVY, PINK, figure, finish


def soc_distributions(d: RunData):
    s = d.sessions
    target = 100 * float(d.cfg["vehicles"]["target_soc"])
    b = d.cfg["vehicles"]["arrival_soc_beta"]
    fig, ax = figure(size=(9.5, 4.6))
    bins = np.arange(0, 101, 2)
    ax.hist(100 * s["soc_arrival"], bins=bins, color=BLUE, alpha=0.55, label="Arrival SOC (sampled)")
    ax.hist(100 * s["soc_end"], bins=bins, color=GREEN, alpha=0.75, label="End SOC (simulated)")
    ax.axvline(target, color="#000000", ls="--", lw=1.1, label=f"Target SOC ({target:.0f} %)")
    reached = 100 * s["target_reached"].mean()
    ax.set_xlim(0, 100)
    ax.set_xlabel("State of charge (%)")
    ax.set_ylabel("Sessions (count per 2 % bin)")
    ax.set_title(f"Arrival vs end SOC, reference run ({reached:.1f} % reach the target)")
    ax.legend(loc="upper left")
    finish(fig, [f"Arrival SOC ~ Beta({b['a']}, {b['b']}): placeholder assumption. Sessions below target "
                 "at departure ran out of plug-in time or were curtailed."])
    return fig


def duration_vs_energy(d: RunData):
    s = d.sessions
    hi = float(np.percentile(s["duration_min"], 99.5))
    fig, ax = figure(size=(9.5, 5.2))
    for flag, c, lab, z in ((False, GREEN, "Not curtailed", 2), (True, PINK, "Curtailed by the grid cap", 3)):
        m = s["curtailed"] == flag
        ax.scatter(s.loc[m, "duration_min"], s.loc[m, "energy_kwh"], s=6 if not flag else 12, color=c,
                   alpha=0.35 if not flag else 0.9, lw=0, label=f"{lab} ({m.sum():,})", zorder=z, rasterized=True)
    ax.set_xlim(0, hi)
    ax.set_ylim(0, None)
    ax.set_xlabel("Plug-in duration (min)")
    ax.set_ylabel("Delivered energy, grid side (kWh)")
    ax.set_title("Plug-in duration vs delivered energy per session (reference run)")
    ax.legend(loc="upper right", markerscale=2.5)
    finish(fig, [f"x axis cut at the 99.5th percentile ({hi:.0f} min)."])
    return fig


def target_by_hour(d: RunData):
    s = d.sessions
    h = pd.DatetimeIndex(s["start"]).hour
    g = s.groupby(h).agg(n=("session_id", "size"), reached=("target_reached", "mean"),
                         curt=("curtailed", "mean")).reindex(range(24))
    fig, axes = figure(2, 1, size=(10, 6.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    x = np.arange(24) + 0.5
    a = axes[0]
    a.plot(x, 100 * g["reached"], color=GREEN, marker="o", ms=4, label="Reach target SOC")
    a.plot(x, 100 * g["curt"], color=PINK, marker="o", ms=4, label="Curtailed by the grid cap")
    a.set_ylim(0, 100)
    a.set_ylabel("Share of sessions (%)")
    a.set_title("Target SOC attainment and curtailment by arrival hour (reference run)")
    a.legend(loc="upper right", ncols=2)
    axes[1].bar(x, g["n"], width=0.8, color=NAVY)
    axes[1].set_ylabel("Sessions (count)")
    axes[1].grid(axis="x", visible=False)
    hour_axis(axes[1], label="Arrival hour (h)")
    finish(fig)
    return fig
