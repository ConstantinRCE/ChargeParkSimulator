"""Load DC charging curves from the Fastned databank (primary) and, optionally,
an evkx databank into one common schema.

Common schema (CurveSet.meta, one row per curve):
  curve_id, source ("fastned" | "evkx_measured" | "evkx_estimated"), brand,
  model, version, category, battery_kwh, peak_kw, charger_class, n_versions
CurveSet.power_kw: float array (n_curves, 101), DC kW at SOC 0..100 %.
CurveSet.versions: every version represented by a curve (curve_id, brand,
  model, version), used to map external per-model weights onto curves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

N_SOC = 101
SOC_COLS = [f"soc_{i}" for i in range(N_SOC)]
META_COLS = ["curve_id", "source", "brand", "model", "version", "category",
             "battery_kwh", "peak_kw", "charger_class", "n_versions"]


@dataclass
class CurveSet:
    meta: pd.DataFrame
    power_kw: np.ndarray
    versions: pd.DataFrame
    log: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.meta)

    def __post_init__(self):
        if self.power_kw.shape != (len(self.meta), N_SOC):
            raise ValueError(f"power_kw shape {self.power_kw.shape} does not match {len(self.meta)} curves")

    def subset(self, mask) -> "CurveSet":
        mask = np.asarray(mask, bool)
        meta = self.meta[mask].reset_index(drop=True)
        return CurveSet(meta, self.power_kw[mask], self.versions[self.versions.curve_id.isin(meta.curve_id)]
                        .reset_index(drop=True), list(self.log))

    @staticmethod
    def concat(sets: list["CurveSet"]) -> "CurveSet":
        return CurveSet(pd.concat([s.meta for s in sets], ignore_index=True),
                        np.vstack([s.power_kw for s in sets]),
                        pd.concat([s.versions for s in sets], ignore_index=True),
                        [line for s in sets for line in s.log])


def norm_name(s: pd.Series) -> pd.Series:
    """Normalize brand / model names for matching."""
    return s.astype(str).str.lower().str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def _wide_curves(path: Path, id_col: str) -> pd.DataFrame:
    w = pd.read_csv(path)
    missing = [c for c in [id_col] + SOC_COLS if c not in w.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing[:5]}{'...' if len(missing) > 5 else ''}")
    return w[[id_col] + SOC_COLS]


# =============================================================================
# Fastned
# =============================================================================
def load_fastned(cfg: dict) -> CurveSet:
    fc = cfg["curves"]["fastned"]
    d = Path(fc["dir"])
    v = pd.read_csv(d / "out" / "vehicles_databank.csv")
    w = _wide_curves(d / "out" / "curves_wide.csv", "vehicle_id")
    log = [f"Fastned: {len(v)} versions in databank"]

    v = v[v["has_curve"] == 1]
    log.append(f"  {len(v)} with a curve")
    v = v[v["battery_kwh_from_name"].notna()]
    log.append(f"  {len(v)} with battery_kwh_from_name")
    v = v[v["category"].isin(fc["categories"])]
    log.append(f"  {len(v)} in categories {fc['categories']}")
    v = v.merge(w, on="vehicle_id", how="inner", validate="one_to_one").copy()

    # one curve per curve_group_id: the version whose battery is closest to the
    # group median (ties: lowest vehicle_id). Siblings share the measured curve.
    med = v.groupby("curve_group_id")["battery_kwh_from_name"].transform("median")
    v = v.assign(_dist=(v["battery_kwh_from_name"] - med).abs())
    keep = v.sort_values(["curve_group_id", "_dist", "vehicle_id"]).drop_duplicates("curve_group_id")
    keep = keep.sort_values("vehicle_id").reset_index(drop=True)
    n_grp = v.groupby("curve_group_id").size()
    multi_cap = v.groupby("curve_group_id")["battery_kwh_from_name"].nunique()
    log.append(f"  {len(keep)} distinct curves after keeping one per curve_group_id "
               f"({int((multi_cap > 1).sum())} groups list different batteries; median kept)")

    rep_id = keep.set_index("curve_group_id")["vehicle_id"]
    versions = pd.DataFrame({
        "curve_id": "fastned:" + v["curve_group_id"].map(rep_id).astype(str),
        "brand": v["brand"].to_numpy(), "model": v["model"].to_numpy(), "version": v["version"].to_numpy(),
    })
    meta = pd.DataFrame({
        "curve_id": "fastned:" + keep["vehicle_id"].astype(str),
        "source": "fastned",
        "brand": keep["brand"], "model": keep["model"], "version": keep["version"],
        "category": keep["category"],
        "battery_kwh": keep["battery_kwh_from_name"].astype(float),
        "peak_kw": keep[SOC_COLS].max(axis=1).astype(float),
        "charger_class": keep["charger_class_norm"].astype(str),
        "n_versions": keep["curve_group_id"].map(n_grp).astype(int),
    })
    return CurveSet(meta[META_COLS], keep[SOC_COLS].to_numpy(float), versions, log)


# =============================================================================
# evkx (optional)
# =============================================================================
def _truthy(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().isin(["true", "1", "yes", "y", "t"])


def load_evkx(cfg: dict) -> CurveSet:
    ec = cfg["curves"]["evkx"]
    c = ec["columns"]
    d = Path(ec["dir"])
    if not d.exists():
        raise FileNotFoundError(f"evkx is enabled but {d} does not exist")
    v = pd.read_csv(d / ec["vehicles_file"])
    w = _wide_curves(d / ec["curves_file"], c["id"])
    log = [f"evkx: {len(v)} rows"]
    required = [c[k] for k in ("id", "brand", "model", "peak_kw", "capacity_kwh", "curve_quality", "estimated")]
    missing = [x for x in required if x not in v.columns]
    if missing:
        raise ValueError(f"{d / ec['vehicles_file']}: missing columns {missing}; adjust curves.evkx.columns")

    v = v[v[c["curve_quality"]].astype(str).str.strip().str.lower() == "ok"]
    log.append(f"  {len(v)} with curve_quality ok")
    est = _truthy(v[c["estimated"]])
    if not ec["include_estimated"]:
        v, est = v[~est], est[~est]
        log.append(f"  {len(v)} measured (estimated dropped)")
    v = v[pd.to_numeric(v[c["capacity_kwh"]], errors="coerce").notna()]
    v = v[pd.to_numeric(v[c["peak_kw"]], errors="coerce") <= ec["max_peak_kw"]]
    log.append(f"  {len(v)} with capacity and peak_kw <= {ec['max_peak_kw']}")
    v = v.merge(w, on=c["id"], how="inner")
    cat = v[c["category"]] if c.get("category") in v.columns else pd.Series("unknown", index=v.index)
    keep = cat.isin(list(cfg["curves"]["fastned"]["categories"]) + ["unknown"])
    v, cat = v[keep], cat[keep]
    log.append(f"  {len(v)} with a curve in the configured categories (or no category)")
    est = _truthy(v[c["estimated"]])
    meta = pd.DataFrame({
        "curve_id": "evkx:" + v[c["id"]].astype(str),
        "source": np.where(est, "evkx_estimated", "evkx_measured"),
        "brand": v[c["brand"]], "model": v[c["model"]],
        "version": v[c["version"]] if c.get("version") in v.columns else "",
        "category": cat,
        "battery_kwh": pd.to_numeric(v[c["capacity_kwh"]]).astype(float),
        "peak_kw": pd.to_numeric(v[c["peak_kw"]]).astype(float),
        "charger_class": v[c["charger_class"]].astype(str) if c.get("charger_class") in v.columns else "",
        "n_versions": 1,
    }).reset_index(drop=True)
    versions = meta[["curve_id", "brand", "model", "version"]].copy()
    return CurveSet(meta[META_COLS], v[SOC_COLS].to_numpy(float), versions, log)


# =============================================================================
# Combined
# =============================================================================
def load_curves(cfg: dict) -> CurveSet:
    """All eligible curves: Fastned, plus evkx if enabled (Fastned wins when the
    same brand/model exists in both)."""
    fast = load_fastned(cfg)
    if not cfg["curves"]["evkx"]["enabled"]:
        return fast
    ev = load_evkx(cfg)
    key = lambda m: norm_name(m["brand"]) + "|" + norm_name(m["model"])
    dup = key(ev.meta).isin(set(key(fast.meta)) | set(key(fast.versions)))
    ev = ev.subset(~dup.to_numpy())
    ev.log.append(f"  {int(dup.sum())} evkx curves dropped (brand/model already in Fastned); {len(ev)} added")
    out = CurveSet.concat([fast, ev])
    if out.meta["curve_id"].duplicated().any():
        raise ValueError("duplicate curve_id after combining sources")
    return out
