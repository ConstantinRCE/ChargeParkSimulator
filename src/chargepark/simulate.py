"""Minute-resolution charging simulation with site load management.

Power convention (documented assumption):
  * Vehicle curves are DC power at the connector, P_dc(SOC), linearly
    interpolated between integer SOC points (0..100 %).
  * Requested P_dc = min(charger_kw, P_dc(SOC) x derating(month), power that
    reaches target SOC within the minute).
  * Grid draw = P_dc / charger_efficiency. The grid cap and load management
    act on grid power; the allocated grid power x charger_efficiency is the
    DC power delivered.
  * SOC gain per minute = P_dc x vehicle_efficiency x dt / battery_kwh.
Loads are grid side, kW, 1-min averages. Session energies are grid side
(energy into the battery = grid energy x charger_eff x vehicle_eff).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numba import njit

from .config import LM_MODES
from .io import year_minutes, year_start
from .curves import N_SOC
from .vehicles import Fleet


# =============================================================================
# Compiled core
# =============================================================================
@njit(cache=True)
def allocate(req, n, cap, mode, out):
    """Distribute `cap` among the first n requests (in arrival order) into out.
    mode 0 equal_share (water-filling), 1 proportional, 2 fifo.
    Returns True if the cap was binding."""
    tot = 0.0
    for i in range(n):
        tot += req[i]
    if tot <= cap:
        for i in range(n):
            out[i] = req[i]
        return False
    if mode == 1:
        f = cap / tot
        for i in range(n):
            out[i] = req[i] * f
    elif mode == 2:
        rem = cap
        for i in range(n):
            a = min(req[i], rem)
            out[i] = a
            rem -= a
    else:
        # serve the smallest requests first; each gets min(request, equal share
        # of what is left), so unused share flows to the larger requests
        order = np.argsort(req[:n])
        rem = cap
        k = n
        for j in range(n):
            i = order[j]
            a = min(req[i], rem / k)
            out[i] = a
            rem -= a
            k -= 1
    return True


@njit(cache=True)
def curve_power(curves, v, soc):
    """Linear interpolation of curve v on the integer SOC grid 0..100 %."""
    x = soc * 100.0
    if x <= 0.0:
        return curves[v, 0]
    if x >= 100.0:
        return curves[v, 100]
    i = int(x)
    f = x - i
    return curves[v, i] + f * (curves[v, i + 1] - curves[v, i])


@njit(cache=True)
def _simulate(start_min, end_min, charger_kw, cap_kwh, vtype, curves, soc0, target_soc,
              charger_eff, vehicle_eff, derate_day, grid_cap_kw, n_minutes, mode, max_active,
              trace_off, tr_p, tr_req, tr_soc):
    """trace_off[s] >= 0: write session s's minute t (t - start_min[s] rows
    after trace_off[s]) to tr_p (allocated grid kW), tr_req (requested grid
    kW) and tr_soc (SOC at the start of the minute)."""
    n_s = start_min.shape[0]
    load = np.zeros(n_minutes)
    n_active = np.zeros(n_minutes, dtype=np.int32)
    bind = np.zeros(n_minutes, dtype=np.bool_)
    e_grid = np.zeros(n_s)
    soc = soc0.copy()
    binding = 0
    active = np.empty(max_active, dtype=np.int64)
    req = np.zeros(max_active)
    alloc = np.zeros(max_active)
    n_act = 0
    nxt = 0
    dt_h = 1.0 / 60.0

    for t in range(n_minutes):
        # drop finished sessions, keeping arrival order
        w = 0
        for j in range(n_act):
            s = active[j]
            if end_min[s] > t:
                active[w] = s
                w += 1
        n_act = w
        # add arrivals
        while nxt < n_s and start_min[nxt] <= t:
            if end_min[nxt] > t:
                active[n_act] = nxt
                n_act += 1
            nxt += 1
        n_active[t] = n_act
        if n_act == 0:
            continue
        derate = derate_day[t // 1440]
        # requested grid power
        for j in range(n_act):
            s = active[j]
            if soc[s] >= target_soc:
                req[j] = 0.0
                continue
            p_dc = min(charger_kw[s], curve_power(curves, vtype[s], soc[s]) * derate)
            p_dc = min(p_dc, (target_soc - soc[s]) * cap_kwh[s] / (dt_h * vehicle_eff))
            req[j] = max(p_dc, 0.0) / charger_eff
        if allocate(req, n_act, grid_cap_kw, mode, alloc):
            binding += 1
            bind[t] = True
        tot = 0.0
        for j in range(n_act):
            s = active[j]
            p = alloc[j]
            o = trace_off[s]
            if o >= 0:
                k = o + t - start_min[s]
                tr_p[k] = p
                tr_req[k] = req[j]
                tr_soc[k] = soc[s]
            tot += p
            e_grid[s] += p * dt_h
            soc[s] += p * charger_eff * vehicle_eff * dt_h / cap_kwh[s]
        load[t] = tot
    return load, e_grid, soc, binding, n_active, bind


# =============================================================================
# Python wrapper
# =============================================================================
@dataclass
class SessionArrays:
    """Sessions as minute offsets from the start of the modeled year.
    end_min is exclusive and clipped to n_minutes. Sorted by start_min."""
    start_min: np.ndarray
    end_min: np.ndarray
    charger_kw: np.ndarray
    n_minutes: int
    year: int | None = None       # enables month-dependent derating

    def __post_init__(self):
        self.start_min = np.ascontiguousarray(self.start_min, dtype=np.int64)
        self.end_min = np.ascontiguousarray(self.end_min, dtype=np.int64)
        self.charger_kw = np.ascontiguousarray(
            np.broadcast_to(np.asarray(self.charger_kw, dtype=float), self.start_min.shape))
        if np.any(np.diff(self.start_min) < 0):
            raise ValueError("sessions must be sorted by start_min")
        if np.any(self.end_min <= self.start_min):
            raise ValueError("every session needs end_min > start_min")

    def __len__(self) -> int:
        return len(self.start_min)

    @classmethod
    def from_frame(cls, sessions: pd.DataFrame, year: int) -> "SessionArrays":
        t0 = year_start(year)
        n_min = year_minutes(year)
        start = ((sessions["start"] - t0) / pd.Timedelta(minutes=1)).to_numpy().astype(np.int64)
        end = np.minimum(start + sessions["duration_min"].to_numpy().astype(np.int64), n_min)
        return cls(start, end, sessions["charger_kw"].to_numpy(), n_min, year)

    def max_concurrency(self) -> int:
        ev = np.concatenate([np.ones(len(self), np.int64), -np.ones(len(self), np.int64)])
        tm = np.concatenate([self.start_min, self.end_min])
        order = np.lexsort((ev, tm))           # at equal time, departures (-1) first
        return int(np.cumsum(ev[order]).max()) if len(self) else 0


@dataclass
class SimResult:
    load_kw: np.ndarray        # per minute, grid side
    energy_kwh: np.ndarray     # per session, grid side
    soc_end: np.ndarray        # per session
    binding_minutes: int
    n_active: np.ndarray | None = None      # plugged-in sessions per minute
    binding: np.ndarray | None = None       # bool per minute: cap binding
    trace: pd.DataFrame | None = None       # per-minute detail of traced sessions


def _trace_layout(sa: "SessionArrays", trace) -> tuple[np.ndarray, np.ndarray]:
    """Row offsets per session (-1 = not traced) and the traced session ids."""
    off = np.full(len(sa), -1, np.int64)
    if trace is None:
        return off, np.zeros(0, np.int64)
    ids = np.flatnonzero(np.asarray(trace, bool)) if np.asarray(trace).dtype == bool         else np.unique(np.asarray(trace, np.int64))
    dur = sa.end_min[ids] - sa.start_min[ids]
    off[ids] = np.concatenate([[0], np.cumsum(dur)[:-1]]) if len(ids) else []
    return off, ids


def derating_by_day(sa: SessionArrays, cfg: dict) -> np.ndarray:
    """Curve power factor per simulated day (temperature derating by month)."""
    n_days = -(-sa.n_minutes // 1440)
    td = cfg["vehicles"]["temperature_derating"]
    if not td["enabled"]:
        return np.ones(n_days)
    if sa.year is None:
        raise ValueError("temperature derating needs SessionArrays.year")
    months = pd.date_range(year_start(sa.year), periods=n_days, freq="D").month.to_numpy()
    return np.asarray(td["monthly_factor"], float)[months - 1]


def simulate(sa: SessionArrays, fleet: Fleet, curves_kw: np.ndarray, cfg: dict,
             grid_cap_kw: float | None = None, lm_mode: str | None = None,
             trace=None) -> SimResult:
    """Run one simulation. curves_kw: (types, 101) DC kW, rows indexed by
    fleet.vtype. grid_cap_kw defaults to cfg; pass np.inf for no cap.
    trace: session ids (or a bool mask) whose per-minute requested power,
    allocated power and SOC are returned in SimResult.trace."""
    if len(fleet) != len(sa):
        raise ValueError(f"fleet has {len(fleet)} vehicles for {len(sa)} sessions")
    curves_kw = np.ascontiguousarray(np.atleast_2d(curves_kw), dtype=float)
    if curves_kw.shape[1] != N_SOC:
        raise ValueError(f"curves need {N_SOC} SOC points, got {curves_kw.shape[1]}")
    if len(fleet) and (fleet.vtype.min() < 0 or fleet.vtype.max() >= len(curves_kw)):
        raise ValueError("fleet.vtype out of range for the curve matrix")
    cap = cfg["site"]["grid_cap_kw"] if grid_cap_kw is None else grid_cap_kw
    mode = LM_MODES[lm_mode or cfg["site"]["lm_mode"]]
    off, ids = _trace_layout(sa, trace)
    n_tr = int((sa.end_min[ids] - sa.start_min[ids]).sum())
    tr_p, tr_req, tr_soc = (np.full(n_tr, np.nan) for _ in range(3))
    load, e, soc, bind, n_active, bind_mask = _simulate(
        sa.start_min, sa.end_min, sa.charger_kw,
        np.ascontiguousarray(fleet.capacity_kwh, float), np.ascontiguousarray(fleet.vtype, np.int64),
        curves_kw, np.ascontiguousarray(fleet.soc0, float),
        float(cfg["vehicles"]["target_soc"]), float(cfg["site"]["charger_efficiency"]),
        float(cfg["vehicles"]["vehicle_efficiency"]), derating_by_day(sa, cfg),
        float(cap), int(sa.n_minutes), mode, max(1, sa.max_concurrency()),
        off, tr_p, tr_req, tr_soc,
    )
    tr = None
    if trace is not None:
        dur = sa.end_min[ids] - sa.start_min[ids]
        sid = np.repeat(ids, dur)
        minute = sa.start_min[sid] + (np.arange(n_tr) - off[sid])
        tr = pd.DataFrame({"session_id": sid, "minute": minute, "power_kw": tr_p,
                           "requested_kw": tr_req, "soc": tr_soc})
    return SimResult(load, e, soc, int(bind), n_active, bind_mask, tr)
