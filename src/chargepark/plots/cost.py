"""05 Cost: procurement cost and the price the station actually pays."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np

from .common import hour_axis
from .data import RunData
from .load import MONTHS
from .style import BLUE, GREEN, NAVY, PINK, PURPLE, figure, finish, seq_cmap, tint


def monthly_cost(d: RunData):
    q = d.qh
    m = q.groupby(q.index.month)[["cost_da_eur", "cost_id_eur", "cost_ppa_eur"]].sum().reindex(range(1, 13),
                                                                                             fill_value=0) / 1000
    parts = [("cost_ppa_eur", GREEN, "PPA"), ("cost_da_eur", BLUE, "Day-ahead (DA)"), ("cost_id_eur", PURPLE,
                                                                                    "Intraday (ID)")]
    if not m["cost_id_eur"].abs().gt(1e-9).any():
        parts = parts[:2]
    fig, ax = figure(size=(10, 4.8))
    x = np.arange(12)
    pos, neg = np.zeros(12), np.zeros(12)
    for col, c, lab in parts:
        v = m[col].to_numpy()
        base = np.where(v >= 0, pos, neg)
        ax.bar(x, v, bottom=base, color=c, width=0.7, label=lab, edgecolor="white", linewidth=1)
        pos += np.clip(v, 0, None)
        neg += np.clip(v, None, 0)
    ax.axhline(0, color="#777777", lw=0.8)
    ax.set_xticks(x, MONTHS)
    ax.grid(axis="x", visible=False)
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy cost (kEUR)")
    p = d.cfg["procurement"]
    ax.set_title(f"Monthly energy cost by channel, reference run (total {m.to_numpy().sum():,.1f} kEUR, "
                 f"mode {p['mode']})")
    ax.legend(loc="upper right", ncols=3)
    finish(fig, [f"PPA {100 * p['share_ppa']:.0f} % at {p['ppa_price_eur_mwh']:.0f} EUR/MWh (placeholder)."]
           + d.notes(prices=True))
    return fig


def weighted_price(d: RunData):
    q = d.qh
    g = q.groupby(q.index.month)
    e = g["energy_kwh"].sum()
    lw = (q["energy_kwh"] * q["da_price"]).groupby(q.index.month).sum() / e.replace(0, np.nan)
    base = g["da_price"].mean()
    allin = g["cost_eur"].sum() / (e / 1000).replace(0, np.nan)
    x = np.arange(1, 13)
    fig, ax = figure(size=(10, 4.8))
    ax.plot(x, base.reindex(x), color=NAVY, marker="o", ms=4, label="Baseload average DA price")
    ax.plot(x, lw.reindex(x), color=BLUE, marker="o", ms=4, label="Load-weighted DA price")
    ax.plot(x, allin.reindex(x), color=GREEN, marker="o", ms=4, label="Average procurement cost incl. PPA")
    ax.set_xticks(x, MONTHS)
    ax.set_xlabel("Month")
    ax.set_ylabel("Price (EUR/MWh)")
    tot_lw = float((q["energy_kwh"] * q["da_price"]).sum() / q["energy_kwh"].sum())
    ax.set_title(f"Load-weighted vs baseload DA price per month (year: {tot_lw:.1f} vs "
                 f"{q['da_price'].mean():.1f} EUR/MWh)")
    ax.legend(loc="best")
    finish(fig, ["Load-weighted above baseload means the station charges in more expensive hours than average."]
           + d.notes(prices=True))
    return fig


def daily_profile(d: RunData):
    q = d.qh
    slot = q.index.hour * 4 + q.index.minute // 15
    load = (q["energy_kwh"] * 4).groupby(slot)
    price = q["da_price"].groupby(slot)
    x = np.arange(96) / 4 + 0.125
    fig, axes = figure(2, 1, size=(10, 6.8), sharex=True)
    for ax, g, c, ylab, lab in ((axes[0], load, GREEN, "Grid load (kW, 15-min mean)", "station load"),
                                (axes[1], price, BLUE, "DA price (EUR/MWh)", "DA price")):
        ax.fill_between(x, g.quantile(0.1), g.quantile(0.9), color=tint(c, 0.75), lw=0,
                        label=f"P10 to P90 over days")
        ax.plot(x, g.mean(), color=c, lw=1.8, label=f"Mean {lab}")
        ax.set_ylabel(ylab)
        ax.legend(loc="upper left", ncols=2)
    axes[1].axhline(0, color="#777777", lw=0.8)
    axes[0].set_ylim(0, None)
    axes[0].set_title("Average daily profile: station load (top) and DA price (bottom)")
    hour_axis(axes[1])
    finish(fig, d.notes(prices=True))
    return fig


def load_vs_price(d: RunData):
    q = d.qh
    fig, ax = figure(size=(9.5, 5.5))
    kw = q["energy_kwh"].to_numpy() * 4
    p = q["da_price"].to_numpy()
    lo, hi = np.percentile(p, [0.2, 99.8])
    hb = ax.hexbin(p, kw, gridsize=(60, 30), extent=(lo, hi, 0, max(kw.max(), 1)), mincnt=1,
                   cmap=seq_cmap(NAVY), norm=mcolors.LogNorm(), linewidths=0, rasterized=True)
    cb = fig.colorbar(hb, ax=ax, pad=0.015)
    cb.set_label("Quarter hours (count, log scale)")
    cb.outline.set_visible(False)
    ax.axvline(0, color="#777777", lw=0.8)
    ax.set_xlabel("DA price (EUR/MWh)")
    ax.set_ylabel("Grid load (kW, 15-min mean)")
    r = np.corrcoef(p, kw)[0, 1] if kw.std() > 0 else np.nan
    ax.set_title(f"Load vs price per quarter hour (correlation {r:.2f})")
    finish(fig, [f"Price axis cut at the 0.2 and 99.8 percentiles. Includes quarter hours with zero load."]
           + d.notes(prices=True))
    return fig
