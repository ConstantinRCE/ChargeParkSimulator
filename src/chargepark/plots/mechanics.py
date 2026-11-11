"""02 Model mechanics: what happens inside one day and inside one session."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .common import cap_line, hour_axis
from .data import RunData
from .style import GREEN, NAVY, WITHOUT_CAP, apply_style, finish, tight, seq_cmap, tint


def example_day(d: RunData, date):
    """Session Gantt colored by delivered power, above the stacked power per session."""
    import matplotlib.pyplot as plt
    day = pd.Timestamp(date).normalize()
    nxt = day + pd.Timedelta(days=1)
    info = d.example_dates.set_index("date").loc[day]
    tr = d.trace[(d.trace["timestamp"] >= day) & (d.trace["timestamp"] < nxt)]
    sess = d.sessions.set_index("session_id")
    ids = sess.loc[tr["session_id"].unique()].sort_values("start").index.to_numpy()
    row = {s: i for i, s in enumerate(ids)}
    minute = ((tr["timestamp"] - day) / pd.Timedelta(minutes=1)).to_numpy().astype(int)
    mat = np.full((len(ids), 1440), np.nan)
    mat[tr["session_id"].map(row).to_numpy().astype(int), minute] = tr["power_kw"].to_numpy()

    apply_style()
    n = len(ids)
    fig = plt.figure(figsize=(11, 8.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 0.018], wspace=0.02,
                          height_ratios=[max(1.0, min(1.6, n / 40)), 1])
    a1 = fig.add_subplot(gs[0, 0])
    a2 = fig.add_subplot(gs[1, 0], sharex=a1)
    cax = fig.add_subplot(gs[0, 1])
    x = np.arange(1441) / 60
    vmax = max(np.nanmax(mat) if n else 1.0, 1.0)
    cmap = seq_cmap(NAVY).with_extremes(bad=(1, 1, 1, 0))     # not plugged in: blank
    im = a1.pcolormesh(x, np.arange(n + 1), np.ma.masked_invalid(mat), cmap=cmap, vmin=0, vmax=vmax,
                       shading="flat", rasterized=True)
    a1.set_ylim(max(n, 1), 0)
    if n == 0:
        a1.text(12, 0.5, "No sessions on this day", ha="center", va="center", color="#555555")
    a1.set_ylabel("Session (sorted by arrival)")
    a1.grid(axis="y", visible=False)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Delivered power, grid side (kW)")
    cb.outline.set_visible(False)
    a1.set_title(f"{info['label']}: {day:%a %d %b %Y}" + chr(10), loc="left")
    a1.text(0, 1.01, info["reason"][:1].upper() + info["reason"][1:], transform=a1.transAxes, fontsize=8, color="#555555",
            va="bottom")

    # stacked power per session (alternating shades only separate neighbouring bands)
    p = np.nan_to_num(mat)
    t = np.arange(1440) / 60
    cum = np.zeros(1440)
    shades = (GREEN, tint(GREEN, 0.45))
    for i in range(n):
        top = cum + p[i]
        if p[i].any():
            a2.fill_between(t, cum, top, color=shades[i % 2], lw=0, step="post", rasterized=True)
        cum = top
    a2.fill_between([], [], color=GREEN, label="Power per session (stacked, alternating shades)")
    l = d.load.loc[day:nxt - pd.Timedelta(minutes=1)]
    a2.step(t, l["load_uncapped_kw"].to_numpy(), where="post", color=WITHOUT_CAP, lw=1.0,
            label="Total without cap")
    cap_line(a2, d.cap_kw)
    a2.set_ylim(0, max(float(l["load_uncapped_kw"].max()), d.cap_kw) * 1.08)
    a2.set_ylabel("Grid power (kW, 1 min)")
    a2.legend(loc="upper left", ncols=3)
    hour_axis(a2)
    a1.tick_params(labelbottom=False)
    finish(fig, [f"Reference run {d.meta['reference_run']}. Sessions plugged in before midnight are shown from 00:00."])
    return fig


def pick_example_sessions(d: RunData, n_curtailed: int = 3, n_total: int = 6) -> pd.DataFrame:
    """Deterministic pick: the traced sessions that lost most energy to the cap,
    then the uncurtailed ones with the most energy."""
    s = d.sessions[d.sessions["session_id"].isin(d.trace["session_id"].unique())].copy()
    s["lost"] = s["energy_uncapped_kwh"] - s["energy_kwh"]
    cur = s[s["curtailed"]].sort_values(["lost", "session_id"], ascending=[False, True]).head(n_curtailed)
    rest = s[~s["curtailed"]].sort_values(["energy_kwh", "session_id"], ascending=[False, True])
    return pd.concat([cur, rest.head(n_total - len(cur))])


def requested_vs_delivered(d: RunData):
    import matplotlib.pyplot as plt
    pick = pick_example_sessions(d)
    names = d.catalog.set_index("rep_index")["name"] if d.catalog is not None else None
    apply_style()
    n = max(len(pick), 1)
    cols = 3 if n > 4 else 2
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(11.5, 3.4 * rows + 0.6), squeeze=False)
    target = 100 * float(d.cfg["vehicles"]["target_soc"])
    for ax, (_, s) in zip(axes.ravel(), pick.iterrows()):
        tr = d.trace[d.trace["session_id"] == s["session_id"]].sort_values("timestamp")
        soc = 100 * tr["soc"].to_numpy()
        ax.plot(soc, tr["requested_kw"], color=NAVY, lw=1.6, label="Requested (vehicle curve and charger)")
        ax.plot(soc, tr["power_kw"], color=GREEN, lw=1.6, ls="--", label="Delivered (after load management)")
        ax.axvline(target, color="#777777", ls=":", lw=1.0, label=f"Target SOC ({target:.0f} %)")
        nm = names.get(s["rep_index"], "") if names is not None else ""
        state = (f"curtailed, {s['lost']:.1f} kWh lost" if s["curtailed"] else "not curtailed")
        ax.set_title(f"{pd.Timestamp(s['start']):%d %b %H:%M}, {s['duration_min']:.0f} min, {state}",
                     fontsize=8.5)
        ax.text(0.02, 0.03, f"{nm}\n{s['battery_kwh']:.0f} kWh, SOC {100 * s['soc_arrival']:.0f} to "
                            f"{100 * s['soc_end']:.0f} %", transform=ax.transAxes, fontsize=7, color="#555555")
        ax.set_xlabel("State of charge (%)")
        ax.set_ylabel("Grid power (kW)")
        ax.set_ylim(0, None)
    for ax in axes.ravel()[len(pick):]:
        ax.set_visible(False)
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncols=3, bbox_to_anchor=(0.005, 0.965))
    n_cur = int(pick["curtailed"].sum())
    fig.suptitle(f"Requested vs delivered power over SOC, {len(pick)} sessions from the example days "
                 f"({n_cur} curtailed)", x=0.01, ha="left", fontweight="bold", fontsize=10)
    notes = ["Requested = min(charger, vehicle curve(SOC), power to reach target) / charger efficiency. "
             "One point per minute."]
    if n_cur < 2:
        notes.append("Fewer than two curtailed sessions exist on the example days of this run.")
    finish(fig, notes)
    tight(fig, rect=(0, 0.05, 1, 0.93))
    return fig
