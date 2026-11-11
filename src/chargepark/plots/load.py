"""03 Load: station load over time, heatmaps, duration curve, cap statistics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import cap_line, day_axis, heatmap, hour_axis, month_axis
from .data import RunData, day_matrix
import matplotlib.colors as mcolors

from .style import (BLUE, GREEN, MISSING, NAVY, PINK, WITH_CAP, WITHOUT_CAP, figure, finish, seq_cmap,
                    tight, tint)

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def _two_series(ax, idx, capped, uncapped, lw=0.7):
    ax.plot(idx, uncapped, color=WITHOUT_CAP, lw=lw, alpha=0.75, label="Without cap", rasterized=True)
    ax.plot(idx, capped, color=WITH_CAP, lw=lw, label="With cap (load management)", rasterized=True)


def load_year_month(d: RunData):
    l15 = d.load_15
    fig, axes = figure(2, 1, size=(11, 7.5))
    a = axes[0]
    day = l15.index.normalize()
    dm = l15.groupby(day).agg(mean=("load_kw", "mean"), peak=("load_kw", "max"),
                              peak_u=("load_uncapped_kw", "max"))
    a.fill_between(dm.index, 0, dm["mean"], color=tint(WITH_CAP, 0.55), lw=0, step="post",
                   label="Daily mean with cap")
    a.step(dm.index, dm["peak_u"], where="post", color=WITHOUT_CAP, lw=0.8, label="Daily peak without cap")
    a.step(dm.index, dm["peak"], where="post", color=WITH_CAP, lw=0.9, label="Daily peak with cap")
    cap_line(a, d.cap_kw)
    month_axis(a)
    a.set_ylim(0, max(float(dm["peak_u"].max()), d.cap_kw) * 1.15)
    a.set_ylabel("Grid load (kW, 15-min mean)")
    a.set_title(f"Station load, reference run {d.meta['reference_run']}: year (daily peak and mean of "
                f"the 15-min load) and busiest month")
    a.legend(loc="upper right", ncols=4)
    mon = l15["cap_minutes"].groupby(l15.index.month).sum()
    m = int(mon.idxmax()) if mon.max() > 0 else int(l15["load_kw"].groupby(l15.index.month).sum().idxmax())
    sl = l15[l15.index.month == m]
    a = axes[1]
    _two_series(a, sl.index, sl["load_kw"], sl["load_uncapped_kw"], lw=0.8)
    cap_line(a, d.cap_kw)
    day_axis(a)
    a.set_ylabel("Grid load (kW, 15-min mean)")
    a.set_title(f"{MONTHS[m - 1]} {d.year} (month with the most minutes at cap)", fontweight="normal")
    a.legend(loc="upper right", ncols=3)
    finish(fig, ["15-min means hide 1-min peaks above the cap that load management removes; "
                 "see the example days at 1-min resolution."])
    return fig


def load_example_days(d: RunData):
    import matplotlib.pyplot as plt
    ex = d.example_dates
    n = len(ex)
    cols = 2
    rows = int(np.ceil(n / cols))
    fig, axes = figure(rows, cols, size=(11.5, 2.6 * rows + 0.8), sharey=True, squeeze=False)
    t = np.arange(1440) / 60
    for ax, (_, r) in zip(axes.ravel(), ex.iterrows()):
        day = r["date"]
        l = d.load.loc[day:day + pd.Timedelta(days=1) - pd.Timedelta(minutes=1)]
        ax.plot(t[:len(l)], l["load_uncapped_kw"], color=WITHOUT_CAP, lw=0.7, alpha=0.8, label="Without cap",
                rasterized=True)
        ax.plot(t[:len(l)], l["load_kw"], color=WITH_CAP, lw=0.8, label="With cap", rasterized=True)
        cap_line(ax, d.cap_kw)
        hour_axis(ax, label="")
        ax.set_title(f"{day:%a %d %b}: {r['label']}", fontsize=8.5)
        ax.set_ylabel("Grid load (kW, 1 min)")
    for ax in axes.ravel()[n:]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Time of day (h)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper right", ncols=3)
    fig.suptitle("Station load on the example days, 1-min resolution", x=0.01, ha="left",
                 fontweight="bold", fontsize=10)
    finish(fig)
    tight(fig, rect=(0, 0.02, 1, 0.95))
    return fig


def load_heatmap(d: RunData):
    mat = day_matrix(d.load_15["load_kw"], 96, "mean", d.dst, d.year)
    fig, ax = figure(size=(10, 6.5))
    heatmap(fig, ax, mat, seq_cmap(GREEN), "Mean grid load (kW, 15 min)", vmin=0, vmax=d.cap_kw)
    ax.set_title(f"Station load with cap by day and quarter hour (reference run {d.meta['reference_run']})")
    finish(fig, [d.dst.note()])
    return fig


def cap_share_heatmap(d: RunData):
    mat = day_matrix(d.load_15["cap_minutes"] / 15 * 100, 96, "mean", d.dst, d.year)
    fig, ax = figure(size=(10, 6.5))
    cm = mcolors.LinearSegmentedColormap.from_list("cap", [tint(PINK, 0.7), PINK, NAVY]).with_extremes(
        under="#FFFFFF", bad=MISSING)
    heatmap(fig, ax, mat, cm, "Minutes with the cap binding (% of the quarter hour)", vmin=1e-6, vmax=100)
    tot = int(d.load["cap_binding"].sum())
    ax.set_title(f"Share of minutes at the grid cap ({tot:,} min = {tot / 60:,.1f} h in the year)")
    finish(fig, [d.dst.note()])
    return fig


def load_duration(d: RunData):
    l15 = d.load_15
    fig, axes = figure(1, 2, size=(11, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    for ax, top in zip(axes, (None, 200)):
        for col, c, lab in (("load_uncapped_kw", WITHOUT_CAP, "Without cap"),
                            ("load_kw", WITH_CAP, "With cap (load management)")):
            q = np.sort(l15[col].to_numpy())[::-1]
            h = np.arange(1, len(q) + 1) / 4
            if top:
                q, h = q[: top * 4], h[: top * 4]
            ax.plot(h, q, color=c, lw=1.6, label=lab)
        cap_line(ax, d.cap_kw)
        ax.set_xlabel("Hours per year at or above the load (h)")
        ax.set_ylabel("Grid load (kW, 15-min mean)")
        ax.set_xlim(0, None)
        ax.set_ylim(0, None)
    axes[0].set_title("Load duration curve, full year")
    axes[1].set_title(f"Top {200} h", fontweight="normal")
    for ax in axes:
        ax.legend(loc="upper right")
    finish(fig)
    return fig


def cap_monthly(d: RunData):
    l = d.load
    hrs = l["cap_binding"].groupby(l.index.month).sum().reindex(range(1, 13), fill_value=0) / 60
    s = d.sessions
    lost = (s["energy_uncapped_kwh"] - s["energy_kwh"]).groupby(pd.DatetimeIndex(s["start"]).month).sum()
    lost = lost.reindex(range(1, 13), fill_value=0) / 1000
    fig, axes = figure(2, 1, size=(10, 6), sharex=True)
    x = np.arange(12)
    axes[0].bar(x, hrs, color=PINK, width=0.7)
    axes[0].set_ylabel("Cap binding (h)")
    axes[0].set_title(f"Grid cap per month: hours binding ({hrs.sum():,.1f} h) and curtailed energy "
                      f"({lost.sum():,.2f} MWh)")
    axes[1].bar(x, lost, color=NAVY, width=0.7)
    axes[1].set_ylabel("Energy lost to cap (MWh)")
    axes[1].set_xticks(x, MONTHS)
    axes[1].set_xlabel("Month")
    for ax in axes:
        ax.grid(axis="x", visible=False)
    finish(fig, ["Energy lost = energy without cap minus energy with cap per session, by arrival month."])
    return fig


def occupancy_heatmap(d: RunData):
    mat = day_matrix(d.load["n_active_sessions"].astype(float), 24, "max", d.dst, d.year)
    fig, ax = figure(size=(10, 6.5))
    heatmap(fig, ax, mat, seq_cmap(BLUE), "Max. sessions plugged in at the same time (count)", vmin=0)
    ax.set_title("Occupancy: maximum simultaneously plugged-in sessions per hour")
    finish(fig, [d.dst.note(), "Sessions that reached the target SOC still occupy a charger until departure."])
    return fig
