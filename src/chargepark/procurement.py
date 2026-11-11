"""15-min aggregation and energy procurement cost (energy only: no grid fees,
levies or taxes)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def to_quarter_hours(load_kw: np.ndarray, index_15: pd.DatetimeIndex) -> pd.Series:
    """1-min grid-side load (kW) -> energy per 15 min (kWh)."""
    if len(load_kw) != 15 * len(index_15):
        raise ValueError(f"{len(load_kw)} minutes do not match {len(index_15)} quarter hours")
    return pd.Series(load_kw.reshape(-1, 15).sum(axis=1) / 60.0, index=index_15)


def profile_forecast(e_mwh: pd.Series) -> np.ndarray:
    """Average profile by (weekday/weekend, quarter hour of day).

    NOTE: computed in-sample from the same year it is applied to, i.e. the
    buyer knows the average profile perfectly. Only the deviation from that
    profile goes to ID, so ID volumes understate real forecast error."""
    idx = e_mwh.index
    key = [idx.day_of_week >= 5, idx.hour * 4 + idx.minute // 15]
    return e_mwh.groupby(key).transform("mean").to_numpy()


def price_energy(e_qh_kwh: pd.Series, prices: pd.DataFrame, cfg: dict) -> dict[str, float]:
    """Cost of buying the 15-min energy series. Returns EUR totals per channel
    and the volumes (MWh) bought in each."""
    if not e_qh_kwh.index.equals(prices.index):
        raise ValueError("energy and price series must share the same 15-min index")
    p = cfg["procurement"]
    e = e_qh_kwh.to_numpy() / 1000.0
    da = prices["da"].to_numpy()
    if p["mode"] != "da_only" and "id" not in prices:
        raise ValueError(f"procurement.mode {p['mode']!r} needs an ID price column")
    idp = prices["id"].to_numpy() if "id" in prices else np.zeros_like(da)
    s_ppa, ppa_price = p["share_ppa"], p["ppa_price_eur_mwh"]

    if p["mode"] == "da_only":
        v_da, v_id = (1 - s_ppa) * e, np.zeros_like(e)
    elif p["mode"] == "fixed_split":
        v_da, v_id = p["share_da"] * e, p["share_id"] * e
    else:
        resid = (1 - s_ppa) * e
        v_da = profile_forecast(pd.Series(resid, index=e_qh_kwh.index))
        v_id = resid - v_da                     # negative = sold back at ID price
    v_ppa = s_ppa * e

    c_da, c_id, c_ppa = float(v_da @ da), float(v_id @ idp), float(v_ppa.sum() * ppa_price)
    return {
        "cost_eur": c_da + c_id + c_ppa,
        "cost_da_eur": c_da, "cost_id_eur": c_id, "cost_ppa_eur": c_ppa,
        "vol_da_mwh": float(v_da.sum()), "vol_id_net_mwh": float(v_id.sum()),
        "vol_id_abs_mwh": float(np.abs(v_id).sum()), "vol_ppa_mwh": float(v_ppa.sum()),
    }
