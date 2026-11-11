"""
charging_park_cost.py

Electricity cost model for a DC / HPC charging park, based on session data
(start date, start time, plug-in duration) without metered energy.

Pipeline
  1. Load sessions (and optionally prices). With --demo, synthetic data is used.
  2. Monte Carlo loop, per run:
       a. sample a vehicle type and an arrival SOC for every session
       b. minute-by-minute simulation (compiled with Numba) using SOC-dependent
          charging curves, a target SOC, and a grid-connection cap enforced by
          load management
       c. the same simulation without the cap, to quantify the cap's effect
  3. Aggregate to 15-min intervals and price with DA / ID / PPA shares
  4. Print a summary, write CSVs and a figure

IMPORTANT: every vehicle, SOC and price parameter in CONFIG is an illustrative
placeholder chosen for a first test. Replace them with your own assumptions
before drawing conclusions from the results.

Usage
  python charging_park_cost.py --demo
  python charging_park_cost.py --sessions sessions.csv --prices prices.csv
  python charging_park_cost.py --sessions sessions.csv --prices prices.csv --runs 200

Input formats (column names are configurable in CONFIG["columns"])
  sessions.csv : start_date (YYYY-MM-DD), start_time (HH:MM[:SS]), duration_min
                 optional: charger_kw (per-session charger rating)
  prices.csv   : timestamp, da_eur_mwh, id_eur_mwh   (any resolution <= 1 h,
                 forward-filled to 15 min; timestamps must be in the same
                 local time convention as the sessions, see DST note below)

Requirements: numpy, pandas, matplotlib, numba
"""

import argparse
import time

import numpy as np
import pandas as pd
from numba import njit

# =============================================================================
# CONFIG (all values are assumptions, adjust to your case)
# =============================================================================
CONFIG = {
    # --- Site -----------------------------------------------------------------
    "grid_cap_kw": 500.0,          # grid connection limit (grid side)
    "charger_kw_default": 150.0,   # used if sessions have no charger_kw column
    "charger_efficiency": 0.95,    # grid -> battery, assumption
    "lm_mode": "equal_share",      # "equal_share" (water-filling), "proportional", "fifo"

    # --- Vehicles -------------------------------------------------------------
    "target_soc": 0.80,
    # Arrival SOC ~ Beta(a, b). a=2.5, b=7 gives a mean of about 0.26.
    "soc_beta_a": 2.5,
    "soc_beta_b": 7.0,
    # Generic, ILLUSTRATIVE vehicle classes (not specific car models
    # power_frac: fraction of peak DC power at SOC = 0, 10, 20, ..., 100 %
    "vehicles": [
        {"name": "small_400V",   "share": 0.30, "cap_kwh": 50.0, "peak_kw": 100.0,
         "power_frac": [0.80, 1.00, 1.00, 0.90, 0.75, 0.60, 0.50, 0.40, 0.25, 0.15, 0.10]},
        {"name": "mid_400V",     "share": 0.45, "cap_kwh": 75.0, "peak_kw": 150.0,
         "power_frac": [0.80, 1.00, 1.00, 0.95, 0.80, 0.65, 0.50, 0.40, 0.25, 0.15, 0.10]},
        {"name": "premium_800V", "share": 0.25, "cap_kwh": 95.0, "peak_kw": 250.0,
         "power_frac": [0.80, 1.00, 1.00, 1.00, 0.85, 0.70, 0.55, 0.40, 0.25, 0.15, 0.10]},
    ],

    # --- Procurement ----------------------------------------------------------
    # "fixed_split": every kWh is priced as s_da*DA + s_id*ID + s_ppa*PPA
    # "forecast":    PPA covers a fixed share; DA buys an average-profile
    #                forecast (weekday/weekend x quarter-hour); ID balances
    #                the deviation (negative ID volume is sold back at ID price)
    "procurement_mode": "forecast",
    "share_ppa": 0.30,
    "share_da": 0.60,              # only used in fixed_split
    "share_id": 0.10,              # only used in fixed_split
    "ppa_price_eur_mwh": 70.0,     # placeholder

    # --- Monte Carlo ----------------------------------------------------------
    "runs": 50,
    "seed": 42,

    # --- Input column names ---------------------------------------------------
    "columns": {
        "start_date": "start_date",
        "start_time": "start_time",
        "duration_min": "duration_min",
        "charger_kw": "charger_kw",
        "price_ts": "timestamp",
        "price_da": "da_eur_mwh",
        "price_id": "id_eur_mwh",
    },
}

SOC_GRID = np.linspace(0.0, 1.0, 11)
LM_MODES = {"equal_share": 0, "proportional": 1, "fifo": 2}
COLORS = {"green": "#005B4C", "blue": "#0000FF", "navy": "#00002C",
          "purple": "#6544FE", "pink": "#FF7F69", "grey": "#9A9A9A"}


# =============================================================================
# 1. Data loading
# =============================================================================
def load_sessions(path, cfg):
    c = cfg["columns"]
    df = pd.read_csv(path)
    start = pd.to_datetime(df[c["start_date"]].astype(str) + " " + df[c["start_time"]].astype(str))
    out = pd.DataFrame({"start": start, "duration_min": df[c["duration_min"]].astype(float)})
    if c["charger_kw"] in df.columns:
        out["charger_kw"] = df[c["charger_kw"]].astype(float)
    else:
        out["charger_kw"] = cfg["charger_kw_default"]
    out = out.dropna()
    out = out[out["duration_min"] > 0].sort_values("start").reset_index(drop=True)
    return out


def load_prices(path, cfg, index_15min):
    c = cfg["columns"]
    df = pd.read_csv(path, parse_dates=[c["price_ts"]]).set_index(c["price_ts"]).sort_index()
    df = df[[c["price_da"], c["price_id"]]].rename(columns={c["price_da"]: "da", c["price_id"]: "id"})
    df = df[~df.index.duplicated(keep="first")]          # DST duplicates, crude handling
    df = df.reindex(df.index.union(index_15min)).ffill().bfill().reindex(index_15min)
    return df


def make_demo_sessions(year=2025, n_chargers=7, mean_per_day=28, seed=1):
    """Synthetic HPC sessions: daytime-weighted arrivals, ~25-30 min dwell,
    cars leave if all chargers are occupied. For testing only."""
    rng = np.random.default_rng(seed)
    hourly = np.array([0.2, 0.1, 0.1, 0.1, 0.2, 0.4, 0.8, 1.2, 1.4, 1.5, 1.6, 1.8,
                       1.9, 1.8, 1.7, 1.8, 2.0, 2.0, 1.7, 1.3, 1.0, 0.7, 0.5, 0.3])
    hourly = hourly / hourly.sum()
    days = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    rows = []
    for d in days:
        weekend_boost = 1.25 if d.dayofweek >= 4 else 1.0
        n = rng.poisson(mean_per_day * weekend_boost)
        hours = rng.choice(24, size=n, p=hourly)
        mins = hours * 60 + rng.integers(0, 60, size=n)
        durs = np.clip(rng.lognormal(np.log(27), 0.45, size=n), 5, 120).round()
        for m, du in zip(np.sort(mins), durs):
            rows.append((d + pd.Timedelta(minutes=int(m)), du))
    df = pd.DataFrame(rows, columns=["start", "duration_min"]).sort_values("start")
    # respect the number of chargers: a car arriving at a full site leaves
    free_at = np.full(n_chargers, np.datetime64("1970-01-01"), dtype="datetime64[m]")
    keep = []
    for s, du in zip(df["start"].values.astype("datetime64[m]"), df["duration_min"].values):
        i = np.argmin(free_at)
        if free_at[i] <= s:
            free_at[i] = s + np.timedelta64(int(du), "m")
            keep.append(True)
        else:
            keep.append(False)
    df = df[keep].reset_index(drop=True)
    return pd.DataFrame({
        "start_date": df["start"].dt.strftime("%Y-%m-%d"),
        "start_time": df["start"].dt.strftime("%H:%M"),
        "duration_min": df["duration_min"].astype(int),
    })


def make_demo_prices(index_15min, seed=2):
    """Synthetic DA/ID prices (shape only, NOT real market data)."""
    rng = np.random.default_rng(seed)
    h = index_15min.hour.values + index_15min.minute.values / 60
    doy = index_15min.dayofyear.values
    solar_dip = -35 * np.exp(-((h - 13) ** 2) / 8) * (0.5 + 0.5 * np.sin((doy - 80) / 365 * 2 * np.pi))
    evening = 30 * np.exp(-((h - 19) ** 2) / 3)
    morning = 15 * np.exp(-((h - 8) ** 2) / 2)
    base = 85 + 15 * np.cos((doy - 15) / 365 * 2 * np.pi)
    hourly_noise = np.repeat(rng.normal(0, 12, len(h) // 4 + 1), 4)[: len(h)]
    da = base + solar_dip + evening + morning + hourly_noise
    idp = da + rng.normal(0, 15, len(h))
    return pd.DataFrame({"da": da, "id": idp}, index=index_15min)


# =============================================================================
# 2. Core simulation (compiled)
# =============================================================================
@njit(cache=True)
def _allocate(req, n, cap, mode, out):
    """Distribute grid capacity among n active sessions (arrival order).
    Returns True if the cap was binding."""
    tot = 0.0
    for i in range(n):
        tot += req[i]
    if tot <= cap:
        for i in range(n):
            out[i] = req[i]
        return False
    if mode == 1:                       # proportional
        f = cap / tot
        for i in range(n):
            out[i] = req[i] * f
    elif mode == 2:                     # first come, first served
        rem = cap
        for i in range(n):
            a = min(req[i], rem)
            out[i] = a
            rem -= a
    else:                               # equal share with caps (water-filling)
        order = np.argsort(req[:n])
        rem = cap
        k = n
        for j in range(n):
            i = order[j]
            share = rem / k
            if req[i] <= share:
                out[i] = req[i]
                rem -= req[i]
            else:
                out[i] = share
                rem -= share
            k -= 1
    return True


@njit(cache=True)
def simulate(start_min, end_min, charger_kw, cap_kwh, peak_kw, vtype, curves,
             soc_grid, soc0, target_soc, eta, grid_cap_kw, n_minutes, mode):
    """Minute-step simulation. Power values are grid side, kW (1-min average).
    Sessions must be sorted by start_min."""
    n_s = start_min.shape[0]
    load = np.zeros(n_minutes)
    e_grid = np.zeros(n_s)
    soc = soc0.copy()
    binding_minutes = 0

    max_active = 512
    active = np.empty(max_active, dtype=np.int64)
    req = np.zeros(max_active)
    alloc = np.zeros(max_active)
    n_act = 0
    nxt = 0
    dt_h = 1.0 / 60.0

    for t in range(n_minutes):
        # drop sessions that have ended (keep arrival order)
        w = 0
        for j in range(n_act):
            s = active[j]
            if end_min[s] > t:
                active[w] = s
                w += 1
        n_act = w
        # add sessions starting now
        while nxt < n_s and start_min[nxt] <= t:
            if end_min[nxt] > t and n_act < max_active:
                active[n_act] = nxt
                n_act += 1
            nxt += 1
        if n_act == 0:
            continue
        # requested grid power per session
        for j in range(n_act):
            s = active[j]
            if soc[s] >= target_soc:
                req[j] = 0.0
                continue
            p_curve = peak_kw[s] * np.interp(soc[s], soc_grid, curves[vtype[s]])
            p_batt = min(charger_kw[s], p_curve)
            p_to_target = (target_soc - soc[s]) * cap_kwh[s] / dt_h   # do not overshoot target
            p_batt = min(p_batt, p_to_target)
            req[j] = p_batt / eta
        if _allocate(req, n_act, grid_cap_kw, mode, alloc):
            binding_minutes += 1
        # update state
        tot = 0.0
        for j in range(n_act):
            s = active[j]
            p = alloc[j]
            tot += p
            e_grid[s] += p * dt_h
            soc[s] += p * eta * dt_h / cap_kwh[s]
        load[t] = tot
    return load, e_grid, soc, binding_minutes


# =============================================================================
# 3. Monte Carlo + costing
# =============================================================================
def sample_vehicles(n, cfg, rng):
    v = cfg["vehicles"]
    shares = np.array([x["share"] for x in v], dtype=float)
    vtype = rng.choice(len(v), size=n, p=shares / shares.sum())
    cap = np.array([x["cap_kwh"] for x in v])[vtype]
    peak = np.array([x["peak_kw"] for x in v])[vtype]
    soc0 = rng.beta(cfg["soc_beta_a"], cfg["soc_beta_b"], size=n)
    return vtype.astype(np.int64), cap.astype(float), peak.astype(float), soc0


def price_energy(e_qh_kwh, prices, cfg):
    """Returns total cost in EUR and a breakdown dict. e_qh_kwh: Series on 15-min index."""
    da = prices["da"].values
    idp = prices["id"].values
    e = e_qh_kwh.values / 1000.0          # MWh
    ppa = cfg["ppa_price_eur_mwh"]
    if cfg["procurement_mode"] == "fixed_split":
        c_da = (cfg["share_da"] * e * da).sum()
        c_id = (cfg["share_id"] * e * idp).sum()
        c_ppa = cfg["share_ppa"] * e.sum() * ppa
    else:
        s_ppa = cfg["share_ppa"]
        resid = (1 - s_ppa) * e
        idx = e_qh_kwh.index
        key = pd.MultiIndex.from_arrays([idx.dayofweek >= 5, idx.hour * 4 + idx.minute // 15])
        fc = pd.Series(resid, index=key).groupby(level=[0, 1]).transform("mean").values
        id_vol = resid - fc
        c_da = (fc * da).sum()
        c_id = (id_vol * idp).sum()
        c_ppa = s_ppa * e.sum() * ppa
    return c_da + c_id + c_ppa, {"DA": c_da, "ID": c_id, "PPA": c_ppa}


def run(sessions, prices, cfg, n_runs):
    t0_year = sessions["start"].min().normalize().replace(month=1, day=1)
    t_end = t0_year + pd.DateOffset(years=1)
    n_minutes = int((t_end - t0_year) / pd.Timedelta(minutes=1))
    idx_15 = pd.date_range(t0_year, t_end, freq="15min", inclusive="left")

    start_min = ((sessions["start"] - t0_year) / pd.Timedelta(minutes=1)).values.astype(np.int64)
    end_min = np.minimum(start_min + sessions["duration_min"].values.astype(np.int64), n_minutes)
    charger_kw = sessions["charger_kw"].values.astype(float)
    curves = np.array([v["power_frac"] for v in cfg["vehicles"]], dtype=float)
    mode = LM_MODES[cfg["lm_mode"]]
    rng = np.random.default_rng(cfg["seed"])

    if prices is None:
        prices = make_demo_prices(idx_15)

    results, first = [], None
    for r in range(n_runs):
        vtype, cap, peak, soc0 = sample_vehicles(len(sessions), cfg, rng)
        args = (start_min, end_min, charger_kw, cap, peak, vtype, curves, SOC_GRID, soc0,
                cfg["target_soc"], cfg["charger_efficiency"])
        load, e, soc_end, bind = simulate(*args, cfg["grid_cap_kw"], n_minutes, mode)
        load_u, e_u, _, _ = simulate(*args, 1e12, n_minutes, mode)

        e_qh = pd.Series(load.reshape(-1, 15).sum(axis=1) / 60.0, index=idx_15)
        cost, parts = price_energy(e_qh, prices, cfg)
        e_tot = e.sum()
        results.append({
            "run": r,
            "energy_mwh": e_tot / 1000,
            "energy_unconstrained_mwh": e_u.sum() / 1000,
            "energy_lost_pct": 100 * (1 - e_tot / e_u.sum()),
            "sessions_curtailed_pct": 100 * np.mean(e < e_u - 0.01),
            "cap_binding_hours": bind / 60,
            "peak_15min_kw": (e_qh.values * 4).max(),
            "cost_eur": cost,
            "cost_da_eur": parts["DA"], "cost_id_eur": parts["ID"], "cost_ppa_eur": parts["PPA"],
            "avg_cost_eur_mwh": cost / (e_tot / 1000),
            "soc_target_reached_pct": 100 * np.mean(soc_end >= cfg["target_soc"] - 1e-6),
        })
        if r == 0:
            first = {"load": load, "load_u": load_u, "e": e, "e_u": e_u, "idx_15": idx_15,
                     "t0": t0_year, "prices": prices}
    return pd.DataFrame(results), first


# =============================================================================
# 4. Output
# =============================================================================
def plot(res, first, sessions, cfg, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#E6E6E6", "grid.linewidth": 0.6})
    fig, ax = plt.subplots(2, 2, figsize=(13, 8))
    cap = cfg["grid_cap_kw"]

    # (a) busiest week, 1-min load
    daily = pd.Series(first["load_u"]).groupby(np.arange(len(first["load_u"])) // 1440).sum()
    wk_start = int(daily.rolling(7).sum().idxmax()) - 6
    sl = slice(wk_start * 1440, (wk_start + 7) * 1440)
    t = first["t0"] + pd.to_timedelta(np.arange(sl.start, sl.stop), unit="min")
    a = ax[0, 0]
    a.plot(t, first["load_u"][sl], color=COLORS["purple"], lw=0.8, alpha=0.55, label="Without load management")
    a.plot(t, first["load"][sl], color=COLORS["green"], lw=0.9, label="With load management")
    a.axhline(cap, color=COLORS["pink"], lw=1.5, ls="--")
    a.text(t[0], cap * 1.03, f"Grid cap {cap:.0f} kW", color=COLORS["navy"], fontsize=8)
    a.set_title("Busiest week, 1-min site load (run 1)", loc="left", fontweight="bold")
    a.set_ylabel("kW")
    a.legend(loc="upper right", frameon=False, fontsize=8)
    a.tick_params(axis="x", rotation=30)

    # (b) load duration curve, 15-min averages
    a = ax[0, 1]
    for key, col, lab in [("load_u", COLORS["purple"], "Without load management"),
                          ("load", COLORS["green"], "With load management")]:
        q = np.sort(first[key].reshape(-1, 15).mean(axis=1))[::-1]
        a.plot(np.arange(len(q)) / 4, q, color=col, lw=2, label=lab)
    a.axhline(cap, color=COLORS["pink"], lw=1.5, ls="--")
    a.set_xlim(0, 3000)
    a.set_title("Load duration curve, 15-min averages (top 3,000 h)", loc="left", fontweight="bold")
    a.set_xlabel("Hours per year")
    a.set_ylabel("kW")
    a.legend(frameon=False, fontsize=8)

    # (c) energy per session vs dwell time
    a = ax[1, 0]
    d = sessions["duration_min"].values
    a.scatter(d, first["e_u"], s=8, color=COLORS["purple"], alpha=0.25, label="Without load management")
    a.scatter(d, first["e"], s=8, color=COLORS["green"], alpha=0.35, label="With load management")
    a.set_xlim(0, np.percentile(d, 99.5))
    a.set_title("Energy per session vs. plug-in duration (run 1)", loc="left", fontweight="bold")
    a.set_xlabel("Plug-in duration (min)")
    a.set_ylabel("Grid energy (kWh)")
    a.legend(frameon=False, fontsize=8, markerscale=2)

    # (d) Monte Carlo distribution of average cost
    a = ax[1, 1]
    v = res["avg_cost_eur_mwh"]
    a.hist(v, bins=min(25, max(5, len(v) // 2)), color=COLORS["blue"], alpha=0.75,
           edgecolor="white", linewidth=2)
    for p, ls in [(10, ":"), (50, "-"), (90, ":")]:
        x = np.percentile(v, p)
        a.axvline(x, color=COLORS["navy"], ls=ls, lw=1.2)
        a.text(x, a.get_ylim()[1] * 0.95, f" P{p}\n {x:.1f}", fontsize=8, color=COLORS["navy"], va="top")
    a.set_title(f"Average energy cost across {len(v)} Monte Carlo runs", loc="left", fontweight="bold")
    a.set_xlabel("EUR/MWh (energy procurement only)")
    a.set_ylabel("Runs")

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summary(res, sessions, cfg, runtime_s):
    p = lambda col: np.percentile(res[col], [10, 50, 90])
    lines = [
        "",
        "=" * 64,
        f"Sessions: {len(sessions):,} | runs: {len(res)} | runtime: {runtime_s:.1f} s",
        f"Grid cap: {cfg['grid_cap_kw']:.0f} kW | LM mode: {cfg['lm_mode']} | procurement: {cfg['procurement_mode']}",
        "-" * 64,
        f"{'Metric':<34}{'P10':>10}{'P50':>10}{'P90':>10}",
    ]
    rows = [
        ("Energy delivered (MWh)", "energy_mwh", "{:.0f}"),
        ("Energy w/o cap (MWh)", "energy_unconstrained_mwh", "{:.0f}"),
        ("Energy lost to cap (%)", "energy_lost_pct", "{:.1f}"),
        ("Sessions curtailed (%)", "sessions_curtailed_pct", "{:.1f}"),
        ("Cap binding (h/yr)", "cap_binding_hours", "{:.0f}"),
        ("Peak 15-min load (kW)", "peak_15min_kw", "{:.0f}"),
        ("Target SOC reached (%)", "soc_target_reached_pct", "{:.1f}"),
        ("Energy cost (kEUR)", "cost_eur", "{:.1f}"),
        ("Avg energy cost (EUR/MWh)", "avg_cost_eur_mwh", "{:.1f}"),
    ]
    for lab, col, fmt in rows:
        vals = p(col)
        if col == "cost_eur":
            vals = vals / 1000
        lines.append(f"{lab:<34}" + "".join(f"{fmt.format(x):>10}" for x in vals))
    lines.append("=" * 64)
    lines.append("Cost covers energy procurement only (no grid fees, levies, taxes, VAT).")
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sessions", help="sessions CSV")
    ap.add_argument("--prices", help="prices CSV (optional; synthetic prices if omitted)")
    ap.add_argument("--runs", type=int, default=CONFIG["runs"])
    ap.add_argument("--demo", action="store_true", help="use synthetic sessions and prices")
    ap.add_argument("--out", default="results", help="output file prefix")
    args = ap.parse_args()

    cfg = CONFIG
    if args.demo or not args.sessions:
        demo = make_demo_sessions()
        demo.to_csv(f"{args.out}_demo_sessions.csv", index=False)
        sessions = load_sessions(f"{args.out}_demo_sessions.csv", cfg)
        print("Using SYNTHETIC demo sessions and prices.")
    else:
        sessions = load_sessions(args.sessions, cfg)

    prices = None
    if args.prices:
        t0 = sessions["start"].min().normalize().replace(month=1, day=1)
        idx = pd.date_range(t0, t0 + pd.DateOffset(years=1), freq="15min", inclusive="left")
        prices = load_prices(args.prices, cfg, idx)
    else:
        print("No price file given: using SYNTHETIC prices (shape only, not market data).")

    t0 = time.time()
    res, first = run(sessions, prices, cfg, args.runs)
    runtime = time.time() - t0

    res.to_csv(f"{args.out}_montecarlo.csv", index=False)
    summary(res, sessions, cfg, runtime)
    plot(res, first, sessions, cfg, f"{args.out}_overview.png")
    print(f"Wrote {args.out}_montecarlo.csv and {args.out}_overview.png")


if __name__ == "__main__":
    main()
