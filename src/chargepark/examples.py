"""Rule-based selection of example days from the reference run.

Every rule picks exactly one day. Ties are broken by an explicit secondary key
and finally by the earliest date, so the selection is deterministic. Two rules
may pick the same day; each rule keeps its own row in example_days.csv.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RULES = ("peak_load", "most_cap_minutes", "median_energy", "most_expensive",
         "most_negative_prices", "typical_weekday", "typical_weekend")

LABELS = {
    "peak_load": "Highest peak load",
    "most_cap_minutes": "Most minutes at cap",
    "median_energy": "Median-energy day",
    "most_expensive": "Most expensive day",
    "most_negative_prices": "Most negative-price quarter hours",
    "typical_weekday": "Typical weekday",
    "typical_weekend": "Typical weekend day",
}


def _pick(daily: pd.DataFrame, keys: list[str], ascending: list[bool]) -> pd.Timestamp:
    """First day after sorting by keys, then by date (stable, deterministic)."""
    d = daily.reset_index(names="date").sort_values(keys + ["date"], ascending=ascending + [True],
                                                    kind="mergesort")
    return pd.Timestamp(d["date"].iloc[0])


def _typical(profiles: pd.DataFrame, mask: np.ndarray) -> tuple[pd.Timestamp, float]:
    """Day whose 15-min profile has the smallest RMSE to the mean profile of the masked days."""
    sub = profiles[mask]
    rmse = np.sqrt(((sub - sub.mean(axis=0)) ** 2).mean(axis=1))
    r = pd.DataFrame({"rmse": rmse})
    day = _pick(r, ["rmse"], [True])
    return day, float(r.loc[day, "rmse"])


def select_example_days(load_1min: pd.DataFrame, qh: pd.DataFrame) -> pd.DataFrame:
    """load_1min: index timestamp, columns load_kw, load_uncapped_kw, cap_binding.
    qh: index timestamp (15 min), columns energy_kwh, da_price, cost_eur.
    Returns one row per rule: rule, label, date, value, unit, reason."""
    lday = load_1min.index.normalize()
    qday = qh.index.normalize()
    daily = pd.DataFrame({
        "peak_uncapped_kw": load_1min["load_uncapped_kw"].groupby(lday).max(),
        "cap_minutes": load_1min["cap_binding"].astype(int).groupby(lday).sum(),
    })
    q = pd.DataFrame({
        "energy_kwh": qh["energy_kwh"].groupby(qday).sum(),
        "cost_eur": qh["cost_eur"].groupby(qday).sum(),
        "n_neg": (qh["da_price"] < 0).astype(int).groupby(qday).sum(),
        "min_price": qh["da_price"].groupby(qday).min(),
    })
    daily = daily.join(q, how="inner")
    daily["med_dev"] = (daily["energy_kwh"] - daily["energy_kwh"].median()).abs()

    slot = qh.index.hour * 4 + qh.index.minute // 15
    prof = qh["energy_kwh"].groupby([qday, slot]).sum().unstack(fill_value=0.0) * 4   # mean kW
    prof = prof.reindex(daily.index).fillna(0.0)
    weekend = prof.index.day_of_week.to_numpy() >= 5

    rows = []

    def add(rule, day, value, unit, reason):
        rows.append({"rule": rule, "label": LABELS[rule], "date": day.strftime("%Y-%m-%d"),
                     "value": float(value), "unit": unit, "reason": reason})

    d = _pick(daily, ["peak_uncapped_kw", "energy_kwh"], [False, False])
    v = daily.loc[d, "peak_uncapped_kw"]
    add("peak_load", d, v, "kW", f"highest 1-min load without cap ({v:,.0f} kW)")

    d = _pick(daily, ["cap_minutes", "peak_uncapped_kw"], [False, False])
    v = daily.loc[d, "cap_minutes"]
    add("most_cap_minutes", d, v, "min", f"most minutes with the grid cap binding ({v:.0f} min)")

    d = _pick(daily, ["med_dev"], [True])
    v = daily.loc[d, "energy_kwh"]
    add("median_energy", d, v, "kWh",
        f"daily energy closest to the median day ({v:,.0f} kWh vs median {daily['energy_kwh'].median():,.0f} kWh)")

    d = _pick(daily, ["cost_eur"], [False])
    v = daily.loc[d, "cost_eur"]
    add("most_expensive", d, v, "EUR", f"highest daily energy cost ({v:,.0f} EUR)")

    d = _pick(daily, ["n_neg", "min_price"], [False, True])
    v = daily.loc[d, "n_neg"]
    add("most_negative_prices", d, v, "quarter hours",
        f"most quarter hours with a negative DA price ({v:.0f}, minimum {daily.loc[d, 'min_price']:.1f} EUR/MWh)")

    for rule, mask, what in (("typical_weekday", ~weekend, "Mon to Fri"),
                             ("typical_weekend", weekend, "Sat and Sun")):
        if mask.any():
            d, v = _typical(prof, mask)
            add(rule, d, v, "kW", f"15-min load profile closest to the average {what} profile "
                                  f"(RMSE {v:.1f} kW)")
    return pd.DataFrame(rows)
