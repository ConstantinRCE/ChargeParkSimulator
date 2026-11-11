"""Load sessions and prices; build synthetic prices for pipeline tests.

Time convention: everything inside the model is naive local wall-clock time
(site.local_timezone), at minute resolution, for one calendar year.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

_TIME_RE = re.compile(r"^\s*(\d{1,2}):(\d{2})(?::(\d{2}))?\s*$")
_TZ_SUFFIX_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})\s*$")


# =============================================================================
# Time grid helpers
# =============================================================================
def year_start(year: int) -> pd.Timestamp:
    return pd.Timestamp(year=year, month=1, day=1)


def year_minutes(year: int) -> int:
    return int((year_start(year + 1) - year_start(year)) / pd.Timedelta(minutes=1))


def year_index_15(year: int) -> pd.DatetimeIndex:
    return pd.date_range(year_start(year), year_start(year + 1), freq="15min", inclusive="left")


def nonexistent_local_hours(year: int, tz: str) -> pd.DatetimeIndex:
    """Naive hours that do not exist in tz (spring-forward gap)."""
    idx = pd.date_range(year_start(year), year_start(year + 1), freq="h", inclusive="left")
    loc = idx.tz_localize(tz, nonexistent="NaT", ambiguous=np.zeros(len(idx), dtype=bool))
    return idx[loc.isna()]


# =============================================================================
# Sessions
# =============================================================================
@dataclass
class SessionReport:
    path: str
    year: int
    n_read: int
    n_kept: int
    dropped: dict[str, int] = field(default_factory=dict)
    n_wrapped: int = 0            # end_time <= arrival_time, ends next day
    n_truncated: int = 0          # runs past the end of the modeled year
    n_duplicates: int = 0         # identical rows (kept)
    n_in_dst_gap: int = 0         # arrival in a nonexistent local hour (kept)
    dropped_rows: pd.DataFrame | None = None

    @property
    def n_dropped(self) -> int:
        return sum(self.dropped.values())

    def __str__(self) -> str:
        lines = [f"Sessions {self.path}: read {self.n_read:,}, kept {self.n_kept:,}, "
                 f"dropped {self.n_dropped:,} (year {self.year})"]
        for reason, n in self.dropped.items():
            lines.append(f"  dropped {n:,} rows: {reason}")
        lines.append(f"  {self.n_wrapped:,} sessions cross midnight (end_time <= arrival_time)")
        if self.n_truncated:
            lines.append(f"  WARNING {self.n_truncated:,} sessions run past the end of {self.year} "
                         f"and are truncated at year end")
        if self.n_duplicates:
            lines.append(f"  WARNING {self.n_duplicates:,} rows are exact duplicates (kept)")
        if self.n_in_dst_gap:
            lines.append(f"  WARNING {self.n_in_dst_gap:,} arrivals fall in the nonexistent DST "
                         f"hour; check that sessions are local wall-clock time")
        return "\n".join(lines)


def _parse_minute_of_day(s: pd.Series) -> pd.Series:
    """'HH:MM' or 'HH:MM:SS' -> minute of day (seconds floored). NaN if invalid."""
    def one(x):
        if not isinstance(x, str):
            return np.nan
        m = _TIME_RE.match(x)
        if not m:
            return np.nan
        h, mi = int(m.group(1)), int(m.group(2))
        sec = int(m.group(3)) if m.group(3) else 0
        if h > 23 or mi > 59 or sec > 59:
            return np.nan
        return h * 60 + mi
    return s.map(one).astype(float)


def load_sessions(path: str | Path, cfg: dict) -> tuple[pd.DataFrame, SessionReport]:
    """Read and validate a sessions CSV.

    Returns a DataFrame sorted by start with columns
      row (1-based data row in the file), start, end, duration_min, charger_kw, wrapped
    and a SessionReport describing what was dropped and why.
    """
    scfg = cfg["sessions"]
    col = scfg["columns"]
    raw = pd.read_csv(path, dtype=str, keep_default_na=True)
    missing_cols = [col[k] for k in ("date", "arrival_time", "end_time") if col[k] not in raw.columns]
    if missing_cols:
        raise ValueError(f"{path}: missing required columns {missing_cols}; found {list(raw.columns)}")

    n = len(raw)
    date = pd.to_datetime(raw[col["date"]].str.strip(), format="%Y-%m-%d", errors="coerce")
    arr = _parse_minute_of_day(raw[col["arrival_time"]])
    end = _parse_minute_of_day(raw[col["end_time"]])
    if col["charger_kw"] in raw.columns:
        charger = pd.to_numeric(raw[col["charger_kw"]], errors="coerce")
    else:
        charger = pd.Series(float(cfg["site"]["charger_kw"]), index=raw.index)

    # year: configured or inferred from valid dates
    # (majority year; rows from other years are dropped and reported)
    year = scfg["year"]
    if year is None:
        counts = date.dropna().dt.year.value_counts()
        if counts.empty or counts.iloc[0] <= counts.sum() / 2:
            raise ValueError(f"{path}: no single year holds a majority of sessions "
                             f"({counts.to_dict()}); set sessions.year in the config")
        year = int(counts.index[0])

    wrapped = end <= arr
    duration = end - arr + np.where(wrapped, 1440, 0)

    # first failing check wins; order matters
    reason = pd.Series(pd.NA, index=raw.index, dtype="string")
    checks = [
        ("missing value", raw[[col["date"], col["arrival_time"], col["end_time"]]].isna().any(axis=1)),
        ("invalid date (expected YYYY-MM-DD)", date.isna()),
        ("invalid arrival_time (expected HH:MM)", arr.isna()),
        ("invalid end_time (expected HH:MM)", end.isna()),
        ("invalid charger_kw (must be > 0)", ~(charger > 0)),
        (f"date outside modeled year {year}", date.dt.year != year),
    ]
    if scfg["max_duration_min"] is not None:
        checks.append((f"duration > max_duration_min ({scfg['max_duration_min']})",
                       duration > scfg["max_duration_min"]))
    for label, bad in checks:
        reason = reason.mask(reason.isna() & bad.fillna(True).astype(bool), label)
    keep = reason.isna().to_numpy()

    dropped = {lab: int((reason == lab).sum()) for lab, _ in checks if (reason == lab).any()}
    dropped_rows = raw.loc[~keep].assign(reason=reason[~keep], row=np.flatnonzero(~keep) + 1)

    start = date[keep] + pd.to_timedelta(arr[keep], unit="min")
    out = pd.DataFrame({
        "row": np.flatnonzero(keep) + 1,
        "start": start.to_numpy(),
        "end": (start + pd.to_timedelta(duration[keep], unit="min")).to_numpy(),
        "duration_min": duration[keep].astype(np.int64).to_numpy(),
        "charger_kw": charger[keep].astype(float).to_numpy(),
        "wrapped": wrapped[keep].to_numpy(),
    }).sort_values(["start", "row"], kind="stable").reset_index(drop=True)

    gap = nonexistent_local_hours(year, cfg["site"]["local_timezone"])
    rep = SessionReport(
        path=str(path), year=year, n_read=n, n_kept=len(out), dropped=dropped,
        n_wrapped=int(out["wrapped"].sum()),
        n_truncated=int((out["end"] > year_start(year + 1)).sum()),
        n_duplicates=int(raw[keep].duplicated().sum()),
        n_in_dst_gap=int(out["start"].dt.floor("h").isin(gap).sum()),
        dropped_rows=dropped_rows,
    )
    return out, rep


# =============================================================================
# Prices
# =============================================================================
PRODUCTS = ("da", "id")
_OFFSET_RE = re.compile(r"^UTC([+-])(\d{2}):?(\d{2})$")


@dataclass
class SeriesReport:
    """Load report for one price series (DA or ID)."""
    product: str
    path: str
    timestamps_tz: str
    n_read: int
    n_missing_values: int = 0         # NaN / non-numeric values (ignored)
    n_outside_year: int = 0
    n_duplicate_timestamps: int = 0   # averaged (e.g. DST fall-back hour)
    max_fill_steps: int = 0           # longest forward fill in 15-min steps
    n_in_dst_gap: int = 0             # timestamps in the nonexistent local hour

    def __str__(self) -> str:
        lines = [f"  {self.product.upper()} {self.path} (timestamps: {self.timestamps_tz}): "
                 f"read {self.n_read:,} values"]
        if self.n_missing_values:
            lines.append(f"    ignored {self.n_missing_values:,} missing/non-numeric values")
        if self.n_outside_year:
            lines.append(f"    ignored {self.n_outside_year:,} values outside the modeled year")
        if self.n_duplicate_timestamps:
            lines.append(f"    averaged {self.n_duplicate_timestamps:,} duplicate local timestamps")
        lines.append(f"    longest forward fill: {self.max_fill_steps} x 15 min")
        if self.n_in_dst_gap:
            lines.append(f"    WARNING {self.n_in_dst_gap:,} timestamps fall in the nonexistent local DST "
                         f"hour; the series is probably not local time. Check timestamps_tz")
        return "\n".join(lines)


@dataclass
class PriceReport:
    series: dict[str, SeriesReport] = field(default_factory=dict)
    # best DA/ID alignment shift in minutes per season, after conversion
    alignment_shift_min: dict[str, int] = field(default_factory=dict)
    synthetic: bool = False

    @property
    def misaligned(self) -> bool:
        return any(v != 0 for v in self.alignment_shift_min.values())

    def source(self) -> str:
        if self.synthetic:
            return "SYNTHETIC (shape only, NOT market data)"
        parts = [f"{k.upper()} {Path(r.path).name} ({r.timestamps_tz})" for k, r in self.series.items()]
        if "id" not in self.series:
            parts.append("no ID data")
        return " | ".join(parts)

    def __str__(self) -> str:
        if self.synthetic:
            return "Prices: SYNTHETIC (shape only, NOT market data)"
        lines = ["Prices:"] + [str(r) for r in self.series.values()]
        if self.alignment_shift_min:
            al = ", ".join(f"{k} {v:+d} min" for k, v in self.alignment_shift_min.items())
            lines.append(f"  DA/ID alignment check (best shift of ID onto DA): {al}")
            if self.misaligned:
                lines.append("  WARNING DA and ID daily profiles line up best with a nonzero shift; "
                             "check the timestamps_tz of both series")
        return "\n".join(lines)


def parse_tz(spec: str):
    """'local' -> None; 'UTC+01:00' -> fixed offset; anything else -> IANA name."""
    if spec == "local":
        return None
    m = _OFFSET_RE.match(spec.replace(" ", ""))
    if m:
        sign = 1 if m.group(1) == "+" else -1
        return dt.timezone(sign * dt.timedelta(hours=int(m.group(2)), minutes=int(m.group(3))))
    return spec


def _to_local_naive(t: pd.DatetimeIndex, spec: str, local_tz: str) -> pd.DatetimeIndex:
    if t.tz is None:
        tz = parse_tz(spec)
        if tz is None:
            return t
        t = t.tz_localize(tz)
    return t.tz_convert(local_tz).tz_localize(None)


def read_price_series(spec: dict, local_tz: str) -> tuple[pd.Series, int]:
    """Read one price series (EUR/MWh) from NetCDF (.nc) or CSV.
    Returns (series on naive local wall-clock timestamps, n_values_read)."""
    path = Path(spec["path"])
    if path.suffix.lower() in (".nc", ".nc4", ".netcdf"):
        import xarray as xr
        with xr.open_dataset(path, engine="h5netcdf") as ds:
            if spec["variable"] not in ds:
                raise ValueError(f"{path}: no variable {spec['variable']!r}; found {list(ds.data_vars)}")
            var = ds[spec["variable"]].load()
            t = pd.DatetimeIndex(pd.to_datetime(var[spec["timestamp"]].values))
            vals = pd.Series(np.asarray(var.values, dtype=float).ravel())
    else:
        raw = pd.read_csv(path, dtype=str)
        missing = [c for c in (spec["timestamp"], spec["variable"]) if c not in raw.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; found {list(raw.columns)}")
        ts = raw[spec["timestamp"]].astype(str).str.strip()
        t = pd.DatetimeIndex(pd.to_datetime(ts, utc=True) if ts.str.contains(_TZ_SUFFIX_RE).any()
                             else pd.to_datetime(ts))
        vals = pd.to_numeric(raw[spec["variable"]], errors="coerce")
    t = _to_local_naive(t, spec["timestamps_tz"], local_tz)
    return pd.Series(vals.to_numpy(), index=t), len(vals)


def align_to_grid(s: pd.Series, year: int, max_fill: int, rep: SeriesReport,
                  local_tz: str) -> pd.Series:
    """Clean a series and forward-fill it onto the 15-min grid of `year`."""
    bad = s.isna().to_numpy() | s.index.isna()
    rep.n_missing_values = int(bad.sum())
    s = s[~bad]
    gap = nonexistent_local_hours(year, local_tz)
    rep.n_in_dst_gap = int(s.index.floor("h").isin(gap).sum())

    idx15 = year_index_15(year)
    inside = (s.index >= idx15[0]) & (s.index < idx15[-1] + pd.Timedelta(minutes=15))
    rep.n_outside_year = int((~inside).sum())
    s = s[inside]
    s.index = s.index.floor("15min")
    rep.n_duplicate_timestamps = int(s.index.duplicated().sum())
    s = s.groupby(level=0).mean().sort_index()

    if len(s) == 0 or s.index[0] > idx15[0]:
        first = s.index[0] if len(s) else "none"
        raise ValueError(f"{rep.path}: {rep.product.upper()} prices must start at {idx15[0]} "
                         f"(first value: {first})")
    pos = s.index.searchsorted(idx15, side="right") - 1
    steps = ((idx15 - s.index[pos]) / pd.Timedelta(minutes=15)).astype(int)
    rep.max_fill_steps = int(steps.max())
    if rep.max_fill_steps > max_fill:
        worst = idx15[int(np.argmax(steps))]
        raise ValueError(f"{rep.path}: {rep.product.upper()} price gap of {rep.max_fill_steps} x 15 min "
                         f"ending {worst} exceeds prices.max_fill_steps={max_fill}")
    return pd.Series(s.to_numpy()[pos], index=idx15)


def alignment_shift(prices: pd.DataFrame, max_shift_steps: int = 8) -> dict[str, int]:
    """Shift (minutes) of the ID median daily profile that best matches DA,
    separately for winter (Nov-Feb) and summer (Apr-Sep). 0 = aligned.
    A summer-only offset of +-60 min points to a DST convention mismatch."""
    out = {}
    months = prices.index.month
    for season, mask in (("winter", np.isin(months, [11, 12, 1, 2])),
                         ("summer", np.isin(months, [4, 5, 6, 7, 8, 9]))):
        if mask.sum() < 96 * 14:
            continue
        sub = prices[mask]
        slot = sub.index.hour * 4 + sub.index.minute // 15
        prof = sub.groupby(slot).median()
        a, b = prof["da"].to_numpy(), prof["id"].to_numpy()
        if np.std(a) < 1e-9 or np.std(b) < 1e-9:
            continue                     # flat profile: no shape to align
        lags = list(range(-max_shift_steps, max_shift_steps + 1))
        corr = [np.corrcoef(a, np.roll(b, k))[0, 1] for k in lags]
        out[season] = 15 * lags[int(np.nanargmax(corr))]
    return out


def load_prices(cfg: dict, year: int, paths: dict | None = None) -> tuple[pd.DataFrame, PriceReport]:
    """Load DA prices and, if configured, ID prices (EUR/MWh) under prices.da /
    prices.id (`paths` override the configured paths; {year} in a path is
    filled in) and forward-fill them onto the 15-min grid of `year`.
    Returns DataFrame[da(, id)] and a report."""
    from .config import MODES_NEEDING_ID
    pcfg = cfg["prices"]
    local = cfg["site"]["local_timezone"]
    rep = PriceReport()
    cols = {}
    for prod in PRODUCTS:
        spec = dict(pcfg[prod])
        if paths and paths.get(prod):
            spec["path"] = paths[prod]
        if not spec.get("path"):
            if prod == "da":
                raise ValueError("no file configured for DA prices (prices.da.path)")
            continue
        spec["path"] = str(spec["path"]).format(year=year)
        if not Path(spec["path"]).exists():
            raise FileNotFoundError(f"{prod.upper()} price file not found: {spec['path']}")
        s, n = read_price_series(spec, local)
        r = SeriesReport(product=prod, path=spec["path"], timestamps_tz=spec["timestamps_tz"], n_read=n)
        cols[prod] = align_to_grid(s, year, pcfg["max_fill_steps"], r, local)
        rep.series[prod] = r
    mode = cfg["procurement"]["mode"]
    if "id" not in cols and mode in MODES_NEEDING_ID:
        raise ValueError(f"procurement.mode {mode!r} needs ID prices (prices.id.path); "
                         f"use mode 'da_only' without ID data")
    out = pd.DataFrame(cols)
    if "id" in out:
        rep.alignment_shift_min = alignment_shift(out)
    return out, rep


def synthetic_prices(year: int, cfg: dict) -> tuple[pd.DataFrame, PriceReport]:
    """SYNTHETIC DA/ID prices with a plausible daily/seasonal shape.
    NOT market data: for exercising the pipeline only."""
    p = cfg["synthetic_prices"]
    idx = year_index_15(year)
    rng = np.random.default_rng(p["seed"])
    h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    doy = idx.day_of_year.to_numpy()
    summer = 0.5 + 0.5 * np.sin((doy - 80) / 365 * 2 * np.pi)
    da = (p["base_eur_mwh"]
          + p["seasonal_amp_eur_mwh"] * np.cos((doy - 15) / 365 * 2 * np.pi)
          + p["solar_dip_eur_mwh"] * np.exp(-((h - 13) ** 2) / 8) * summer
          + p["evening_peak_eur_mwh"] * np.exp(-((h - 19) ** 2) / 3)
          + p["morning_peak_eur_mwh"] * np.exp(-((h - 8) ** 2) / 2)
          + np.repeat(rng.normal(0, p["hourly_noise_sd_eur_mwh"], len(h) // 4 + 1), 4)[: len(h)])
    idp = da + rng.normal(0, p["id_noise_sd_eur_mwh"], len(h))
    return (pd.DataFrame({"da": da, "id": idp}, index=idx),
            PriceReport(synthetic=True))
