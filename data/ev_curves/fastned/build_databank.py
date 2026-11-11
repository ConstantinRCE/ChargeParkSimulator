"""
build_databank.py

Builds an EV DC charging-curve databank from Fastned's public brand pages
(raw export: raw/fastned_vehicles.csv, raw/fastned_charge_curves.csv,
collected 2026-11-10 from https://www.fastnedcharging.com/en/brands-overview).

Outputs (out/):
  vehicles_databank.csv   one row per vehicle version, metadata + derived fields
  curves_long.csv         vehicle_id x SOC 0..100, power_kw, power_frac, point_type
  curves_wide.csv         vehicle_id rows x 101 SOC columns (kW), for fast array loading
  qa_overview.png         QA charts

Processing rules (all documented in README.md):
  - measured points are kept unchanged
  - internal SOC gaps: linear interpolation          (point_type = "interpolated")
  - below the first measured SOC: hold first value   (point_type = "extrapolated_low")
  - above the last measured SOC: linear ramp to 0 kW at 100 % SOC
                                                     (point_type = "extrapolated_high")
  - battery capacity is parsed from the version name only; never guessed
"""
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path("raw")
OUT = Path("out")
OUT.mkdir(exist_ok=True)
SOC = np.arange(101)

COL = {"green": "#005B4C", "blue": "#0000FF", "navy": "#00002C",
       "purple": "#6544FE", "pink": "#FF7F69", "grey": "#9A9A9A"}


def parse_capacity(version):
    m = re.search(r"(\d+(?:[.,]\d+)?)\s*kWh", str(version), flags=re.I)
    return float(m.group(1).replace(",", ".")) if m else np.nan


def norm_class(label):
    """Normalised charger class (raw label is kept separately)."""
    if pd.isna(label):
        return np.nan
    digits = re.sub(r"\D", "", str(label))
    return f"{digits}+" if "+" in str(label) else digits


def build_curve(pts):
    """pts: DataFrame with soc_pct, power_kw (measured). Returns kW[101], type[101]."""
    s = pts["soc_pct"].to_numpy(int)
    p = pts["power_kw"].to_numpy(float)
    kw = np.full(101, np.nan)
    typ = np.empty(101, dtype=object)
    lo, hi = s.min(), s.max()
    inside = (SOC >= lo) & (SOC <= hi)
    kw[inside] = np.interp(SOC[inside], s, p)
    typ[inside] = "interpolated"
    typ[s] = "measured"
    kw[SOC < lo] = p[0]
    typ[SOC < lo] = "extrapolated_low"
    if hi < 100:
        up = SOC > hi
        kw[up] = p[-1] * (100 - SOC[up]) / (100 - hi)
        typ[up] = "extrapolated_high"
    return kw, typ


def main():
    v = pd.read_csv(RAW / "fastned_vehicles.csv", encoding="utf-8-sig")
    c = pd.read_csv(RAW / "fastned_charge_curves.csv", encoding="utf-8-sig")
    assert len(v) == 647 and len(c) == 26742, "unexpected raw row counts"

    v["battery_kwh_from_name"] = v["version"].map(parse_capacity)
    v["charger_class_raw"] = v["curve_charger_class"]
    v["charger_class_norm"] = v["curve_charger_class"].map(norm_class)

    long_rows, wide = [], {}
    stats = {}
    for vid, pts in c.sort_values(["vehicle_id", "soc_pct"]).groupby("vehicle_id"):
        kw, typ = build_curve(pts)
        peak = pts["power_kw"].max()
        meas = pts.set_index("soc_pct")["power_kw"]
        wide[vid] = kw
        long_rows.append(pd.DataFrame({
            "vehicle_id": vid, "soc_pct": SOC, "power_kw": kw.round(2),
            "power_frac": (kw / peak).round(4), "point_type": typ}))
        # time 10->80 % (only if capacity known); battery-side, ideal conditions
        stats[vid] = {
            "peak_kw_measured": peak,
            "soc_at_peak": int(meas.idxmax()),
            "avg_kw_10_80": float(np.mean(kw[10:80])),
            "n_measured_points": len(pts),
            "n_gap_points_filled": int((typ == "interpolated").sum()),
            "curve_hash": hashlib.md5(np.round(pts["power_kw"].to_numpy(), 1).tobytes()).hexdigest()[:10],
        }

    s = pd.DataFrame.from_dict(stats, orient="index")
    s.index.name = "vehicle_id"
    v = v.merge(s, left_on="vehicle_id", right_index=True, how="left")

    # identical measured curves shared by several versions (e.g. platform siblings)
    grp = v.dropna(subset=["curve_hash"]).groupby("curve_hash")["vehicle_id"]
    v["curve_shared_with_n"] = v["curve_hash"].map(grp.transform("count").groupby(v["curve_hash"]).first()) - 1
    v["curve_group_id"] = v["curve_hash"].map(grp.min())

    v["minutes_10_80"] = np.where(
        v["battery_kwh_from_name"].notna() & v["avg_kw_10_80"].notna(),
        0.70 * v["battery_kwh_from_name"] / v["avg_kw_10_80"] * 60, np.nan).round(1)

    cols = ["vehicle_id", "brand", "model", "version", "category", "connector_type",
            "recommended_charger", "max_charge_speed_kw", "autocharge_capable",
            "has_curve", "charger_class_raw", "charger_class_norm",
            "battery_kwh_from_name", "peak_kw_measured", "soc_at_peak", "avg_kw_10_80",
            "minutes_10_80", "curve_soc_min", "curve_soc_max", "n_measured_points",
            "n_gap_points_filled", "curve_group_id", "curve_shared_with_n", "source_url"]
    v = v[cols].sort_values(["brand", "model", "version"])
    v["avg_kw_10_80"] = v["avg_kw_10_80"].round(1)
    for col in ["soc_at_peak", "curve_soc_min", "curve_soc_max", "n_measured_points",
                "n_gap_points_filled", "curve_group_id", "curve_shared_with_n"]:
        v[col] = v[col].astype("Int64")
    v.to_csv(OUT / "vehicles_databank.csv", index=False)

    pd.concat(long_rows).to_csv(OUT / "curves_long.csv", index=False)
    w = pd.DataFrame(wide).T.round(2)
    w.columns = [f"soc_{i}" for i in SOC]
    w.index.name = "vehicle_id"
    w.to_csv(OUT / "curves_wide.csv")

    plot_qa(v, w)
    report(v)


def plot_qa(v, w):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6})
    cv = v[v.has_curve == 1].set_index("vehicle_id")
    classes = [("50", COL["grey"]), ("150", COL["purple"]), ("300", COL["green"]), ("400", COL["blue"])]
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))

    # (a)-(b): curves by class family
    for a, fam in zip([ax[0, 0], ax[0, 1]], [("50", "150"), ("300", "400")]):
        for cls, col in classes:
            if cls not in fam:
                continue
            ids = cv.index[cv.charger_class_norm.str.replace("+", "", regex=False) == cls]
            for i in ids:
                a.plot(range(101), w.loc[i].values, color=col, lw=0.6, alpha=0.35)
            a.plot([], [], color=col, lw=2, label=f"Class {cls} / {cls}+  (n={len(ids)})")
        hi_med = cv.curve_soc_max.median()
        a.axvspan(hi_med, 100, color="#F2F2F2", zorder=0)
        a.text(hi_med - 1, a.get_ylim()[1] * 0.04, f"beyond {hi_med:.0f} %:\nmostly extrapolated",
               fontsize=7.5, color=COL["navy"], ha="right",
               bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
        a.set_xlabel("SOC (%)")
        a.set_ylabel("Power (kW)")
        a.legend(frameon=False, fontsize=8, loc="upper right")
    ax[0, 0].set_title("Charging curves, Fastned classes 50 and 150", loc="left", fontweight="bold")
    ax[0, 1].set_title("Charging curves, Fastned classes 300 and 400", loc="left", fontweight="bold")

    # (c) peak power distribution
    a = ax[1, 0]
    a.hist(cv.peak_kw_measured, bins=np.arange(0, 425, 25), color=COL["green"], edgecolor="white", linewidth=2)
    a.set_title("Peak measured power per version", loc="left", fontweight="bold")
    a.set_xlabel("kW")
    a.set_ylabel("Versions")

    # (d) measured SOC coverage
    a = ax[1, 1]
    o = cv.sort_values("curve_soc_min")
    a.hlines(range(len(o)), o.curve_soc_min, o.curve_soc_max, color=COL["purple"], lw=1)
    a.axvline(80, color=COL["pink"], ls="--", lw=1.5)
    a.text(79, len(o) * 0.5, "target SOC 80 %", color=COL["navy"], fontsize=8, ha="right",
           bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5))
    a.set_title("Measured SOC range per version (one line each)", loc="left", fontweight="bold")
    a.set_xlabel("SOC (%)")
    a.set_ylabel("Versions (sorted)")
    a.set_xlim(0, 100)

    fig.tight_layout()
    fig.savefig(OUT / "qa_overview.png", dpi=150)
    plt.close(fig)


def report(v):
    cv = v[v.has_curve == 1]
    print("versions:", len(v), "| with curve:", len(cv))
    print("with curve and capacity from name:", cv.battery_kwh_from_name.notna().sum())
    print("charger class (norm):", cv.charger_class_norm.value_counts().to_dict())
    print("distinct measured curves:", cv.curve_group_id.nunique(),
          "| versions sharing a curve with another:", (cv.curve_shared_with_n > 0).sum())
    print("SOC min  median %.0f | SOC max median %.0f | versions measured to >=80 %%: %d"
          % (cv.curve_soc_min.median(), cv.curve_soc_max.median(), (cv.curve_soc_max >= 80).sum()))
    print("peak kW  P10/P50/P90:", np.percentile(cv.peak_kw_measured, [10, 50, 90]).round(0))
    print("10-80 min P10/P50/P90:", np.nanpercentile(cv.minutes_10_80, [10, 50, 90]).round(0))
    print("versions with zero-kW points:",
          pd.read_csv(OUT / "curves_long.csv").query("point_type=='measured' and power_kw<=0").vehicle_id.nunique())


if __name__ == "__main__":
    main()
