"""Plot catalog: every plot with its section, file name and function, in report order."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from . import cost, inputs, load, mechanics, montecarlo, sessions
from .data import RunData

SECTIONS = {
    "01": "inputs", "02": "mechanics", "03": "load", "04": "sessions", "05": "cost", "06": "montecarlo",
}
TITLES = {
    "01": "Inputs", "02": "Model mechanics", "03": "Load", "04": "Sessions", "05": "Cost", "06": "Monte Carlo",
}


@dataclass
class PlotSpec:
    section: str                      # "01".."06"
    name: str                         # file name without extension, e.g. 03_load_heatmap
    func: Callable
    kwargs: dict = field(default_factory=dict)

    def make(self, d: RunData):
        return self.func(d, **self.kwargs)


def build_catalog(d: RunData) -> list[PlotSpec]:
    P = PlotSpec
    specs = [
        P("01", "01_arrivals_heatmap", inputs.arrivals_heatmap),
        P("01", "01_duration_hist", inputs.duration_hist),
        P("01", "01_arrivals_hour_of_week", inputs.arrivals_hour_of_week),
        P("01", "01_representative_curves", inputs.representative_curves),
        P("01", "01_representative_shares", inputs.representative_shares),
        P("01", "01_price_heatmap", inputs.price_heatmap),
        P("01", "01_price_timeseries", inputs.price_timeseries),
    ]
    for _, r in d.example_dates.iterrows():
        specs.append(P("02", f"02_example_day_{r['rules'][0]}", mechanics.example_day, {"date": r["date"]}))
    specs += [
        P("02", "02_requested_vs_delivered", mechanics.requested_vs_delivered),
        P("03", "03_load_year_month", load.load_year_month),
        P("03", "03_load_example_days", load.load_example_days),
        P("03", "03_load_heatmap", load.load_heatmap),
        P("03", "03_cap_share_heatmap", load.cap_share_heatmap),
        P("03", "03_load_duration", load.load_duration),
        P("03", "03_cap_monthly", load.cap_monthly),
        P("03", "03_occupancy_heatmap", load.occupancy_heatmap),
        P("04", "04_soc_distributions", sessions.soc_distributions),
        P("04", "04_duration_vs_energy", sessions.duration_vs_energy),
        P("04", "04_target_by_hour", sessions.target_by_hour),
        P("05", "05_monthly_cost", cost.monthly_cost),
        P("05", "05_weighted_price", cost.weighted_price),
        P("05", "05_daily_profile", cost.daily_profile),
        P("05", "05_load_vs_price", cost.load_vs_price),
        P("06", "06_kpi_distributions", montecarlo.kpi_distributions),
        P("06", "06_convergence", montecarlo.convergence),
        P("06", "06_profile_band", montecarlo.profile_band),
    ]
    return specs


def section_key(s: str) -> str:
    """'3', '03', 'load' or '03_load' -> '03'."""
    s = s.strip().lower()
    for k, v in SECTIONS.items():
        if s in (k, k.lstrip("0"), v, f"{k}_{v}"):
            return k
    raise ValueError(f"unknown section {s!r}; use one of {', '.join(f'{k} ({v})' for k, v in SECTIONS.items())}")
