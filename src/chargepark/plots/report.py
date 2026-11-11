"""Title page and example-day table for the combined report.pdf."""

from __future__ import annotations

import numpy as np

from ..report import METRICS
from .data import RunData
from .style import GREEN, NAVY, TEXT_MUTED, apply_style


def _kv_block(fig, x, y, title, rows, step=0.026):
    fig.text(x, y, title, fontsize=11, fontweight="bold", color=GREEN)
    y -= step * 1.3
    for k, v in rows:
        fig.text(x, y, k, fontsize=8.5, color=TEXT_MUTED)
        fig.text(x + 0.2, y, str(v), fontsize=8.5, color=NAVY)
        y -= step
    return y


def title_page(d: RunData):
    import matplotlib.pyplot as plt
    apply_style()
    fig = plt.figure(figsize=(11.69, 8.27))
    c, m = d.cfg, d.meta
    site, veh, proc = c["site"], c["vehicles"], c["procurement"]
    fig.text(0.05, 0.92, "Charging park electricity cost model", fontsize=20, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.885, f"Run {d.run_id}  |  created {m.get('created', '')}  |  year {d.year}",
             fontsize=10, color=TEXT_MUTED)
    b = veh["arrival_soc_beta"]
    rows = [
        ("Sessions", f"{m.get('n_sessions', len(d.sessions)):,} ({_name(m.get('sessions', ''))})"),
        ("Monte Carlo runs", f"{m.get('runs', len(d.mc))} (reference run {m['reference_run']}: cost closest to median)"),
        ("Grid cap / charger", f"{site['grid_cap_kw']:,.0f} kW / {site['charger_kw']:,.0f} kW DC"),
        ("Load management", site["lm_mode"]),
        ("Charger efficiency", f"{site['charger_efficiency']:.2f}"),
        ("Vehicle curves", f"k = {m.get('k', c['representatives']['k'])}, {m.get('n_curves', '')} Fastned curves"),
        ("Target SOC / arrival SOC", f"{100 * veh['target_soc']:.0f} % / Beta({b['a']}, {b['b']})"),
        ("Procurement", f"{proc['mode']}, PPA {100 * proc['share_ppa']:.0f} % at {proc['ppa_price_eur_mwh']:.0f} EUR/MWh"),
    ]
    for i, part in enumerate(str(m.get("price_source", "")).split(" | ")):
        rows.append(("Prices" if i == 0 else "", part[:55]))
    _kv_block(fig, 0.05, 0.80, "Configuration", rows)

    mc = d.mc
    fig.text(0.55, 0.80, "Headline KPIs (over runs)", fontsize=11, fontweight="bold", color=GREEN)
    y = 0.80 - 0.026 * 1.3
    for i, h in enumerate(("P10", "P50", "P90")):
        fig.text(0.82 + 0.055 * i, y, h, fontsize=8.5, fontweight="bold", color=NAVY, ha="right")
    y -= 0.024
    for label, col, fmt, sc in METRICS:
        if col not in mc:
            continue
        fig.text(0.55, y, label, fontsize=8.5, color=TEXT_MUTED)
        for i, v in enumerate(np.percentile(mc[col], [10, 50, 90]) * sc):
            fig.text(0.82 + 0.055 * i, y, fmt.format(v), fontsize=8.5, color=NAVY, ha="right")
        y -= 0.024

    notes = ["Cost scope: energy procurement only (no grid fees, levies, taxes, VAT).",
             "Placeholder assumptions: arrival SOC distribution, PPA share and PPA price.",
             "Vehicle mix: equal weight per distinct curve, not market shares (unless a weights CSV is set).",
             "Charging curves reflect about 30 C battery conditions (source: Fastned databank)."]
    if d.synthetic:
        notes.insert(0, "PRICES ARE SYNTHETIC TEST DATA, NOT MARKET PRICES. Cost figures are not a real estimate.")
    if proc["mode"] == "da_only":
        notes.append("No ID prices: non-PPA energy at DA in the quarter hour it is used (perfect forecast).")
    y = 0.22
    fig.text(0.05, y, "Notes", fontsize=11, fontweight="bold", color=GREEN)
    for n in notes:
        y -= 0.026
        fig.text(0.05, y, n, fontsize=8.5, color="#B00000" if "SYNTHETIC" in n else TEXT_MUTED)
    return fig


def example_days_page(d: RunData):
    import matplotlib.pyplot as plt
    apply_style()
    fig = plt.figure(figsize=(11.69, 8.27))
    fig.text(0.05, 0.92, "Example days (chosen by rule from the reference run)", fontsize=14,
             fontweight="bold", color=NAVY)
    y = 0.85
    for _, r in d.example_days.iterrows():
        fig.text(0.05, y, f"{r['date']:%a %d %b %Y}", fontsize=9, color=NAVY, fontweight="bold")
        fig.text(0.2, y, r["label"], fontsize=9, color=GREEN)
        fig.text(0.2, y - 0.025, r["reason"], fontsize=8.5, color=TEXT_MUTED)
        y -= 0.07
    return fig


def _name(p: str) -> str:
    return p.replace("\\", "/").rsplit("/", 1)[-1]
