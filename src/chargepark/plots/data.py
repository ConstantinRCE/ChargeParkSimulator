"""Read a results/<run_id>/ folder. Never runs the simulation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..curves import SOC_COLS

REQUIRED = ("config_snapshot.yaml", "run_meta.json", "mc_summary.csv", "load_1min.parquet",
            "qh.parquet", "sessions.parquet", "session_power_examples.parquet", "example_days.csv")


@dataclass
class DSTInfo:
    spring_day: pd.Timestamp | None       # local day with the nonexistent hour
    spring_hours: list[int]               # nonexistent hours on that day
    autumn_day: pd.Timestamp | None       # local day with the repeated hour
    autumn_hours: list[int]

    def note(self) -> str:
        parts = []
        if self.spring_day is not None:
            h = self.spring_hours[0]
            parts.append(f"{self.spring_day:%d %b} {h:02d}:00 to {h + 1:02d}:00 does not exist (grey cells)")
        if self.autumn_day is not None:
            h = self.autumn_hours[0]
            parts.append(f"{self.autumn_day:%d %b} {h:02d}:00 to {h + 1:02d}:00 occurred twice and is shown once "
                         f"(prices averaged)")
        if not parts:
            return ""
        return "Time axis: local wall-clock time, 24 h per day. Daylight saving: " + "; ".join(parts) + "."


def dst_info(year: int, tz: str) -> DSTInfo:
    idx = pd.date_range(f"{year}-01-01", f"{year + 1}-01-01", freq="h", inclusive="left")
    gap = idx[idx.tz_localize(tz, nonexistent="NaT", ambiguous=np.zeros(len(idx), bool)).isna()]
    amb = idx[idx.tz_localize(tz, nonexistent="shift_forward", ambiguous="NaT").isna()]
    first = lambda x: (x[0].normalize(), sorted(set(x.hour))) if len(x) else (None, [])
    return DSTInfo(*first(gap), *first(amb))


class RunData:
    """Lazy access to every result file of one run."""

    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        missing = [f for f in REQUIRED if not (self.run_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"{self.run_dir} is not a complete results folder; missing {missing}")

    def _pq(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.run_dir / name, engine="pyarrow")

    # --- raw files -----------------------------------------------------------
    @cached_property
    def meta(self) -> dict:
        return json.loads((self.run_dir / "run_meta.json").read_text(encoding="utf-8"))

    @cached_property
    def cfg(self) -> dict:
        return yaml.safe_load((self.run_dir / "config_snapshot.yaml").read_text(encoding="utf-8"))

    @cached_property
    def mc(self) -> pd.DataFrame:
        return pd.read_csv(self.run_dir / "mc_summary.csv")

    @cached_property
    def profiles(self) -> pd.DataFrame | None:
        p = self.run_dir / "mc_profiles_15min.parquet"
        return pd.read_parquet(p).set_index("timestamp") if p.exists() else None

    @cached_property
    def load(self) -> pd.DataFrame:
        return self._pq("load_1min.parquet").set_index("timestamp")

    @cached_property
    def qh(self) -> pd.DataFrame:
        return self._pq("qh.parquet").set_index("timestamp")

    @cached_property
    def sessions(self) -> pd.DataFrame:
        return self._pq("sessions.parquet")

    @cached_property
    def trace(self) -> pd.DataFrame:
        return self._pq("session_power_examples.parquet")

    @cached_property
    def example_days(self) -> pd.DataFrame:
        df = pd.read_csv(self.run_dir / "example_days.csv")
        df["date"] = pd.to_datetime(df["date"])
        return df

    @cached_property
    def catalog(self) -> pd.DataFrame | None:
        p = self.run_dir / "catalog_curves.parquet"
        return pd.read_parquet(p) if p.exists() else None

    @cached_property
    def members(self) -> pd.DataFrame | None:
        p = self.run_dir / "curve_members.parquet"
        return pd.read_parquet(p) if p.exists() else None

    # --- derived -------------------------------------------------------------
    @property
    def run_id(self) -> str:
        return self.meta.get("run_id", self.run_dir.name)

    @property
    def year(self) -> int:
        return int(self.meta["year"])

    @property
    def cap_kw(self) -> float:
        return float(self.cfg["site"]["grid_cap_kw"])

    @property
    def synthetic(self) -> bool:
        return bool(self.meta.get("synthetic_prices", False))

    @property
    def has_id(self) -> bool:
        return bool(self.qh["id_price"].notna().any())

    @cached_property
    def dst(self) -> DSTInfo:
        return dst_info(self.year, self.cfg["site"].get("local_timezone", "Europe/Berlin"))

    @cached_property
    def load_15(self) -> pd.DataFrame:
        """15-min mean of the 1-min load (kW) plus minutes at cap per slot."""
        l = self.load
        g = l.groupby(l.index.floor("15min"))
        return pd.DataFrame({"load_kw": g["load_kw"].mean(), "load_uncapped_kw": g["load_uncapped_kw"].mean(),
                             "cap_minutes": g["cap_binding"].sum()})

    @cached_property
    def example_dates(self) -> pd.DataFrame:
        """One row per distinct example date (rules joined), in date order of first rule."""
        ex = self.example_days
        rows = []
        for d, g in ex.groupby("date", sort=False):
            rows.append({"date": d, "rules": list(g["rule"]), "label": " and ".join(g["label"]),
                         "reason": "; ".join(g["reason"])})
        return pd.DataFrame(rows)

    def example(self, rule: str) -> pd.Timestamp | None:
        m = self.example_days["rule"] == rule
        return self.example_days.loc[m, "date"].iloc[0] if m.any() else None

    def curve_kw(self, frame: pd.DataFrame) -> np.ndarray:
        return frame[SOC_COLS].to_numpy(float)

    def notes(self, prices: bool = False, assumptions: bool = False) -> list[str]:
        out = []
        if prices and self.synthetic:
            out.append("Prices are SYNTHETIC test data, not market prices: cost figures are not a real estimate.")
        elif prices:
            out.append(f"Prices: {self.meta.get('price_source', 'DA day-ahead')}. Energy procurement only "
                       "(no grid fees, levies, taxes).")
        if assumptions:
            out.append("Arrival SOC distribution, PPA share and PPA price are placeholder assumptions; "
                       "vehicle mix is equal weight per curve, not market shares.")
        return out


def day_matrix(s: pd.Series, slots_per_day: int, agg: str = "mean", dst: DSTInfo | None = None,
               year: int | None = None) -> pd.DataFrame:
    """Day (rows) x slot of day (columns) matrix of a naive local time series.
    All days of the year are present; the nonexistent spring DST hour is NaN."""
    idx = s.index
    minutes = idx.hour * 60 + idx.minute
    slot = minutes // (1440 // slots_per_day)
    m = s.groupby([idx.normalize(), slot]).agg(agg).unstack()
    m = m.reindex(columns=range(slots_per_day))
    if year is not None:
        m = m.reindex(pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D"))
    if dst is not None and dst.spring_day is not None and dst.spring_day in m.index:
        per_h = slots_per_day // 24
        for h in dst.spring_hours:
            m.loc[dst.spring_day, h * per_h:(h + 1) * per_h - 1] = np.nan
    return m
