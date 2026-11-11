"""01 Inputs: sessions, charging curves, prices."""

from __future__ import annotations

import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from .common import day_axis, heatmap, month_axis
from .data import RunData, day_matrix
from .style import BLUE, CAP_LINE, GREEN, GRID, NAVY, PINK, SERIES, div_cmap, figure, finish, seq_cmap, tint

DAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]


def arrivals_heatmap(d: RunData):
    s = d.sessions
    cnt = pd.Series(1, index=pd.DatetimeIndex(s["start"]))
    mat = day_matrix(cnt, 24, "sum", d.dst, d.year).fillna(0)
    if d.dst.spring_day is not None:
        mat.loc[d.dst.spring_day, d.dst.spring_hours] = np.nan
    fig, ax = figure(size=(10, 6.5))
    heatmap(fig, ax, mat, seq_cmap(NAVY), "Sessions started per hour (count)", vmin=0)
    ax.set_title(f"Session arrivals by day and hour ({len(s):,} sessions, {d.year})")
    finish(fig, [d.dst.note()])
    return fig


def duration_hist(d: RunData):
    dur = d.sessions["duration_min"].to_numpy()
    hi = max(float(np.percentile(dur, 99.5)), 10.0)
    fig, ax = figure(size=(9, 4.5))
    ax.hist(np.clip(dur, 0, hi), bins=np.arange(0, hi + 5, 5), color=GREEN, edgecolor="white", linewidth=0.5,
            label="Sessions")
    med = float(np.median(dur))
    ax.axvline(med, color=CAP_LINE, ls="--", lw=1.1, label=f"Median {med:.0f} min")
    ax.set_xlim(0, hi)
    ax.set_xlabel("Plug-in duration (min), last bin includes longer sessions")
    ax.set_ylabel("Sessions (count per 5 min bin)")
    ax.set_title("Plug-in duration (arrival to departure, not charging time)")
    ax.legend()
    finish(fig, [f"{(dur > hi).sum():,} sessions longer than {hi:.0f} min are in the last bin."])
    return fig


def arrivals_hour_of_week(d: RunData):
    st = pd.DatetimeIndex(d.sessions["start"])
    how = st.day_of_week * 24 + st.hour
    n_weeks = 365.25 / 7
    cnt = np.bincount(how, minlength=168) / n_weeks
    fig, ax = figure(size=(11, 4.2))
    ax.bar(np.arange(168) + 0.5, cnt, width=0.85, color=GREEN, label="Arrivals")
    for k in range(1, 7):
        ax.axvline(24 * k, color="#999999", lw=0.8)
    ax.set_xticks(np.arange(7) * 24 + 12, DAYS)
    ax.set_xticks(np.arange(0, 169, 6), minor=True)
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlim(0, 168)
    ax.set_xlabel("Hour of week (day, 6 h minor ticks)")
    ax.set_ylabel("Mean arrivals per hour (sessions/h)")
    ax.set_title("Arrivals per hour of week, average over the year")
    finish(fig)
    return fig


def _top_reps(d: RunData, n: int) -> pd.DataFrame:
    """Representatives ranked by delivered energy in the reference run."""
    s = d.sessions
    g = s.groupby("rep_index").agg(sessions=("session_id", "size"), energy=("energy_kwh", "sum"))
    cat = d.catalog.set_index("rep_index").join(g).fillna({"sessions": 0, "energy": 0})
    return cat.sort_values(["energy", "share"], ascending=False).head(n)


def _with_kwh(name: str, kwh: float) -> str:
    """Curve names usually contain the battery size already; add it only if not."""
    return name if "kWh" in name else f"{name} ({kwh:.0f} kWh)"


def representative_curves(d: RunData):
    cat, mem = d.catalog, d.members
    charger = float(d.cfg["site"]["charger_kw"])
    target = 100 * float(d.cfg["vehicles"]["target_soc"])
    soc = np.arange(101)
    k = len(cat)
    fig, ax = figure(size=(10, 5.5))
    if k <= 5:
        for i, r in cat.iterrows():
            c = SERIES[i]
            m = mem[mem["rep_index"] == r["rep_index"]]
            for y in d.curve_kw(m):
                ax.plot(soc, y, color=tint(c, 0.7), lw=0.6, rasterized=True)
            ax.plot(soc, d.curve_kw(cat.loc[[i]])[0], color=c, lw=2.2,
                    label=f"{r['name']} ({r['battery_kwh']:.0f} kWh, {100 * r['share']:.0f} % of sessions, "
                          f"{len(m)} members)")
        title = f"Representative charging curves (k = {k}) with cluster members"
    else:
        for y in d.curve_kw(mem):
            ax.plot(soc, y, color=tint(NAVY, 0.8), lw=0.5, rasterized=True)
        ax.plot([], [], color=tint(NAVY, 0.6), lw=1, label=f"All {len(mem)} curves in the catalog")
        top = _top_reps(d, 5)
        for c, (i, r) in zip(SERIES, top.iterrows()):
            ax.plot(soc, d.curve_kw(cat[cat["rep_index"] == i])[0], color=c, lw=2.0,
                    label=_with_kwh(r["name"], r["battery_kwh"]))
        title = (f"Charging curves: {k} representatives (k = all); "
                 f"5 with the most energy in the reference run highlighted")
    ax.axhline(charger, color=CAP_LINE, ls="--", lw=1.2, label=f"Charger limit ({charger:.0f} kW)")
    ax.axvline(target, color="#777777", ls=":", lw=1.2, label=f"Target SOC ({target:.0f} %)")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, None)
    ax.set_xlabel("State of charge (%)")
    ax.set_ylabel("DC power at connector (kW)")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=7)
    finish(fig, ["Source: Fastned charging-curve databank (about 30 C battery conditions). "
                 "Power above the charger limit is cut by the charger."])
    return fig


def representative_shares(d: RunData, n: int = 15):
    s = d.sessions
    g = s.groupby("rep_index").agg(sessions=("session_id", "size"), energy=("energy_kwh", "sum"))
    g = d.catalog.set_index("rep_index")[["name", "battery_kwh"]].join(g).fillna(0)
    g["sessions"] *= 100 / g["sessions"].sum()
    g["energy"] *= 100 / max(g["energy"].sum(), 1e-12)
    g = g.sort_values(["energy", "sessions"], ascending=False)
    top = g.head(n).copy()
    top["label"] = [_with_kwh(a, b) for a, b in zip(top["name"], top["battery_kwh"])]
    if len(g) > n:
        rest = g.iloc[n:]
        top.loc[-1] = {"name": "Other", "battery_kwh": np.nan, "sessions": rest["sessions"].sum(),
                       "energy": rest["energy"].sum(), "label": f"Other {len(rest)} representatives"}
    y = np.arange(len(top))[::-1]
    fig, ax = figure(size=(10, 0.35 * len(top) + 1.8))
    h = 0.38
    ax.barh(y + h / 2, top["sessions"], height=h, color=GREEN, label="Share of sessions")
    ax.barh(y - h / 2, top["energy"], height=h, color=BLUE, label="Share of delivered energy")
    ax.set_yticks(y, top["label"], fontsize=7.5)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Share in the reference run (%)")
    ax.set_title(f"Sessions and energy per representative curve (top {min(n, len(g))} by energy)")
    ax.legend(loc="lower right")
    finish(fig, ["Vehicle sampling weights: equal weight per distinct curve unless a weights CSV is set "
                 "(not market shares)."])
    return fig


def _price_norm(p: np.ndarray):
    lo, hi = np.nanpercentile(p, [0.5, 99.5])
    lo, hi = min(lo, -1.0), max(hi, 1.0)
    return mcolors.TwoSlopeNorm(vmin=lo, vcenter=0.0, vmax=hi)


def price_heatmap(d: RunData):
    p = d.qh["da_price"]
    mat = day_matrix(p, 96, "mean", d.dst, d.year)
    fig, ax = figure(size=(10, 6.5))
    heatmap(fig, ax, mat, div_cmap(), "DA price (EUR/MWh), centered at 0", norm=_price_norm(p.to_numpy()))
    ax.set_title(f"Day-ahead price by day and quarter hour ({d.year})")
    finish(fig, [d.dst.note(), "Color scale clipped at the 0.5 and 99.5 percentiles."] + d.notes(prices=True))
    return fig


def price_timeseries(d: RunData):
    qh = d.qh
    p = qh["da_price"]
    daily = p.groupby(p.index.normalize()).agg(["mean", "min", "max"])
    fig, axes = figure(3, 1, size=(11, 9))
    a = axes[0]
    a.fill_between(daily.index, daily["min"], daily["max"], color=tint(BLUE, 0.75), step="mid",
                   label="Daily min to max (15 min)", rasterized=True)
    a.plot(daily.index, daily["mean"], color=BLUE, lw=1.1, label="Daily mean")
    a.axhline(0, color="#777777", lw=0.8)
    month_axis(a)
    a.set_ylabel("DA price (EUR/MWh)")
    a.set_title(f"Day-ahead price {d.year}: year (daily), month and week (15 min)")
    a.legend(loc="upper right", ncols=2)

    def zoom(ax, start, end, title):
        sl = qh.loc[start:end - pd.Timedelta(minutes=1)]
        ax.step(sl.index, sl["da_price"], where="post", color=BLUE, lw=1.0, label="DA price")
        if d.has_id:
            ax.step(sl.index, sl["id_price"], where="post", color=PINK, lw=0.8, label="ID price")
        ax.axhline(0, color="#777777", lw=0.8)
        day_axis(ax)
        ax.set_ylabel("Price (EUR/MWh)")
        ax.set_title(title, fontweight="normal")
        if d.has_id:
            ax.legend(loc="upper right", ncols=2)

    ref = d.example("most_expensive") or p.idxmax().normalize()
    m0 = ref.replace(day=1)
    zoom(axes[1], m0, m0 + pd.offsets.MonthBegin(1), f"{m0:%B %Y} (month of the most expensive day)")
    wk = d.example("most_negative_prices") or p.idxmin().normalize()
    w0 = wk - pd.Timedelta(days=wk.day_of_week)
    w0 = max(w0, pd.Timestamp(f"{d.year}-01-01"))
    zoom(axes[2], w0, w0 + pd.Timedelta(days=7),
         f"Week from {w0:%a %d %b} (contains the day with the most negative-price quarter hours)")
    for ax in axes:
        ax.grid(True, color=GRID)
    finish(fig, d.notes(prices=True))
    return fig
