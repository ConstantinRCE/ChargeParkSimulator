"""Hand-checkable simulation cases.

Most cases use flat curves (constant DC kW over SOC), efficiencies of 1 and
large batteries, so power stays constant and energy = kW x minutes / 60.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chargepark.config import load_config
from chargepark.curves import N_SOC, load_curves
from chargepark.io import load_sessions
from chargepark.simulate import SessionArrays, allocate, curve_power, simulate
from chargepark.vehicles import Catalog, fixed_fleet, sample_fleet

FASTNED = Path(__file__).resolve().parents[1] / "data" / "ev_curves" / "fastned" / "out"
needs_fastned = pytest.mark.skipif(not (FASTNED / "curves_wide.csv").exists(),
                                   reason="Fastned databank not extracted")


def make_cfg(cap=1e9, ceff=1.0, veff=1.0, mode="equal_share", target=0.8, charger_kw=150.0):
    return load_config(overrides={
        "site": {"grid_cap_kw": cap, "charger_efficiency": ceff, "lm_mode": mode, "charger_kw": charger_kw},
        "vehicles": {"target_soc": target, "vehicle_efficiency": veff}})


def run(cat, vtype, soc0, start, end, cfg, n_minutes, charger_kw=150.0, **kw):
    sa = SessionArrays(start, end, charger_kw, n_minutes=n_minutes)
    return simulate(sa, fixed_fleet(cat, vtype, soc0), cat.power_kw, cfg, **kw)


def alloc(req, cap, mode):
    req = np.asarray(req, float)
    out = np.zeros_like(req)
    binding = allocate(req, len(req), cap, mode, out)
    return out, binding


# ------------------------------------------------------------ allocate()
@pytest.mark.parametrize("mode", [0, 1, 2])
def test_allocate_under_cap_is_passthrough(mode):
    out, b = alloc([50, 60], 200, mode)
    assert not b and np.allclose(out, [50, 60])


def test_allocate_equal_share_water_filling():
    out, b = alloc([120, 60], 150, 0)        # 75 each; B needs 60, A gets the rest
    assert b and np.allclose(out, [90, 60])
    out, _ = alloc([120, 100], 150, 0)       # both above 75
    assert np.allclose(out, [75, 75])
    out, _ = alloc([10, 100, 100], 150, 0)   # 10 + 70 + 70
    assert np.allclose(out, [10, 70, 70])


def test_allocate_proportional():
    out, _ = alloc([120, 60], 150, 1)        # factor 150/180
    assert np.allclose(out, [100, 50])


def test_allocate_fifo_follows_arrival_order():
    out, _ = alloc([120, 60], 150, 2)
    assert np.allclose(out, [120, 30])
    out, _ = alloc([60, 120], 150, 2)
    assert np.allclose(out, [60, 90])


# ------------------------------------------------------------ curve lookup
def test_curve_power_linear_between_integer_soc():
    c = np.zeros((1, N_SOC))
    c[0, 10], c[0, 11] = 100.0, 120.0
    assert curve_power(c, 0, 0.10) == pytest.approx(100.0)
    assert curve_power(c, 0, 0.105) == pytest.approx(110.0)
    assert curve_power(c, 0, 0.1075) == pytest.approx(115.0)
    c[0, 100] = 7.0
    assert curve_power(c, 0, 1.2) == 7.0 and curve_power(c, 0, -0.1) == 0.0


# ------------------------------------------------------------ single session
def test_single_session_no_cap_flat_curve():
    # 100 kW flat, 100 kWh battery, SOC 0.2, 30 min plugged in, no cap:
    # 100 kW x 0.5 h = 50 kWh, SOC 0.2 -> 0.7, target not reached
    cat = Catalog.flat(100, 100)
    r = run(cat, 0, 0.2, [10], [40], make_cfg(), 60)
    assert r.energy_kwh[0] == pytest.approx(50.0)
    assert r.soc_end[0] == pytest.approx(0.7)
    assert np.allclose(r.load_kw[10:40], 100.0)
    assert np.all(r.load_kw[:10] == 0) and np.all(r.load_kw[40:] == 0)
    assert r.binding_minutes == 0


def test_single_session_stops_at_target():
    # needs (0.8 - 0.7) x 100 kWh = 10 kWh; at 100 kW that takes exactly 6 min
    r = run(Catalog.flat(100, 100), 0, 0.7, [0], [30], make_cfg(), 30)
    assert r.energy_kwh[0] == pytest.approx(10.0)
    assert r.soc_end[0] == pytest.approx(0.8)
    assert np.allclose(r.load_kw[:6], 100.0)
    assert np.allclose(r.load_kw[6:], 0.0, atol=1e-9)


def test_charger_limit_and_efficiencies():
    # vehicle could take 250 kW DC, charger delivers 150 kW DC.
    # grid = 150 / 0.95; 12 min -> 30 kWh DC; SOC gain = 30 x 0.9 / 1000
    r = run(Catalog.flat(250, 1000), 0, 0.0, [0], [12], make_cfg(ceff=0.95, veff=0.9), 12)
    assert np.allclose(r.load_kw, 150 / 0.95)
    assert r.energy_kwh[0] == pytest.approx(30 / 0.95)
    assert r.soc_end[0] == pytest.approx(30 * 0.9 / 1000)


def test_vehicle_efficiency_target_reached_exactly():
    # 10 kWh into the cells at veff 0.8 needs 12.5 kWh DC; at 100 kW DC
    # that is 7.5 min -> 7 full minutes + a partial one
    r = run(Catalog.flat(100, 100), 0, 0.7, [0], [30], make_cfg(veff=0.8), 30)
    assert r.soc_end[0] == pytest.approx(0.8)
    assert r.energy_kwh[0] == pytest.approx(12.5)


def test_arrival_above_target_draws_nothing():
    r = run(Catalog.flat(150, 75), 0, 0.85, [0], [30], make_cfg(), 30)
    assert r.energy_kwh[0] == 0.0 and r.load_kw.sum() == 0.0


def test_temperature_derating_by_month():
    # flat 100 kW curve, factor 0.5 in January: 30 min -> 25 kWh in Jan, 50 kWh in Jul
    cfg = make_cfg()
    cfg["vehicles"]["temperature_derating"] = {"enabled": True,
                                               "monthly_factor": [0.5] + [1.0] * 11}
    cat = Catalog.flat(100, 1000)
    jul = int((pd.Timestamp("2025-07-01") - pd.Timestamp("2025-01-01")) / pd.Timedelta("1min"))
    sa = SessionArrays([60, jul], [90, jul + 30], 150.0, n_minutes=365 * 1440, year=2025)
    r = simulate(sa, fixed_fleet(cat, [0, 0], 0.0), cat.power_kw, cfg)
    assert r.energy_kwh == pytest.approx([25.0, 50.0])


# ------------------------------------------------------------ real curve reproduction
@needs_fastned
def test_single_vehicle_uncapped_reproduces_databank_curve(cfg):
    # highest-power Fastned curve, no charger limit, no grid cap, efficiencies 1,
    # charge from 0 to 100 %: the 1-min load must equal the databank curve at the
    # SOC reached at the start of each minute.
    cs = load_curves(cfg)
    i = int(np.argmax(cs.meta["peak_kw"]))
    curve, bat = cs.power_kw[i], cs.meta.loc[i, "battery_kwh"]
    cat = Catalog(curve[None, :], [bat], [1.0], ["test"])
    c = make_cfg(target=1.0, charger_kw=10_000)
    n = 240
    r = run(cat, 0, 0.0, [0], [n], c, n, charger_kw=10_000)
    soc_start = np.r_[0.0, np.cumsum(r.load_kw)[:-1] / 60 / bat]     # SOC at start of each minute
    mask = soc_start < 0.97                                           # before the tail approaches 0 kW
    expected = np.interp(soc_start[mask] * 100, np.arange(N_SOC), curve)
    assert mask.sum() > 10 and soc_start[mask].max() > 0.9
    np.testing.assert_allclose(r.load_kw[mask], expected, rtol=1e-12)
    assert r.load_kw.max() == pytest.approx(curve.max(), rel=0.02)   # peak is reached


# ------------------------------------------------------------ two sessions over the cap
# A: 120 kW, plugged 0..30. B: 60 kW, plugged 10..30. Cap 150 kW.
#   equal_share:  A 90,  B 60   -> A 20 + 30   = 50 kWh,   B 20 kWh
#   proportional: A 100, B 50   -> A 20 + 33.3 = 53.3 kWh, B 16.7 kWh
#   fifo:         A 120, B 30   -> A 20 + 40   = 60 kWh,   B 10 kWh
@pytest.mark.parametrize("mode,pa,pb", [
    ("equal_share", 90.0, 60.0),
    ("proportional", 100.0, 50.0),
    ("fifo", 120.0, 30.0),
])
def test_two_sessions_exceeding_cap(mode, pa, pb):
    cat = Catalog.flat([120, 60], 1000)
    cfg = make_cfg(cap=150, mode=mode)
    r = run(cat, [0, 1], 0.0, [0, 10], [30, 30], cfg, 40)
    assert np.allclose(r.load_kw[:10], 120.0)
    assert np.allclose(r.load_kw[10:30], 150.0)
    assert r.binding_minutes == 20
    assert r.energy_kwh[0] == pytest.approx(120 * 10 / 60 + pa * 20 / 60)
    assert r.energy_kwh[1] == pytest.approx(pb * 20 / 60)
    u = run(cat, [0, 1], 0.0, [0, 10], [30, 30], cfg, 40, grid_cap_kw=np.inf)
    assert u.energy_kwh == pytest.approx([60.0, 20.0])
    assert u.binding_minutes == 0


# ------------------------------------------------------------ midnight / year end
def test_midnight_crossing_session_end_to_end(tmp_path):
    # 23:50 -> 00:20 next day at a flat 60 kW: 10 kWh on day 1, 20 kWh on day 2
    p = tmp_path / "s.csv"
    p.write_text("date,arrival_time,end_time\n2025-03-10,23:50,00:20\n", encoding="utf-8")
    cfg = make_cfg()
    sessions, _ = load_sessions(p, cfg)
    sa = SessionArrays.from_frame(sessions, 2025)
    cat = Catalog.flat(60, 1000)
    r = simulate(sa, fixed_fleet(cat, 0, 0.0), cat.power_kw, cfg)
    day = r.load_kw.reshape(-1, 1440).sum(axis=1) / 60     # kWh per day
    assert day[68] == pytest.approx(10.0)                   # 2025-03-10 is day index 68
    assert day[69] == pytest.approx(20.0)
    assert r.energy_kwh[0] == pytest.approx(30.0)


def test_year_end_session_truncated(tmp_path):
    p = tmp_path / "s.csv"
    p.write_text("date,arrival_time,end_time\n2025-12-31,23:50,00:30\n", encoding="utf-8")
    cfg = make_cfg()
    sessions, rep = load_sessions(p, cfg)
    sa = SessionArrays.from_frame(sessions, 2025)
    assert rep.n_truncated == 1 and sa.end_min[0] == sa.n_minutes == 365 * 1440
    cat = Catalog.flat(60, 1000)
    r = simulate(sa, fixed_fleet(cat, 0, 0.0), cat.power_kw, cfg)
    assert r.energy_kwh[0] == pytest.approx(10.0)


# ------------------------------------------------------------ invariants on real-curve fleets
def random_sessions(n, n_minutes, seed):
    rng = np.random.default_rng(seed)
    start = np.sort(rng.integers(0, n_minutes - 5, n))
    end = np.minimum(start + rng.integers(5, 90, n), n_minutes)
    return SessionArrays(start, end, 150.0, n_minutes)


def real_catalog(cfg):
    if not (FASTNED / "curves_wide.csv").exists():
        pytest.skip("Fastned databank not extracted")
    from chargepark.vehicles import build_catalog
    return build_catalog(cfg, k=12)[0]


@pytest.mark.parametrize("mode", ["equal_share", "proportional", "fifo"])
def test_energy_never_exceeds_capacity_times_soc_gap(mode):
    cfg = load_config(overrides={"site": {"lm_mode": mode}, "vehicles": {"vehicle_efficiency": 0.92}})
    ceff, veff = cfg["site"]["charger_efficiency"], cfg["vehicles"]["vehicle_efficiency"]
    target = cfg["vehicles"]["target_soc"]
    cat = real_catalog(cfg)
    sa = random_sessions(3000, 3 * 1440, seed=1)
    fleet = sample_fleet(len(sa), cat, cfg, np.random.default_rng(2))
    fleet.soc0[:50] = 0.9                                   # some arrive above target
    for cap in (cfg["site"]["grid_cap_kw"], np.inf):
        r = simulate(sa, fleet, cat.power_kw, cfg, grid_cap_kw=cap)
        bound = fleet.capacity_kwh * np.maximum(target - fleet.soc0, 0.0)
        into_cells = r.energy_kwh * ceff * veff
        assert np.all(into_cells <= bound + 1e-9)
        assert np.all(r.energy_kwh[:50] == 0)
        assert np.all(r.soc_end <= np.maximum(target, fleet.soc0) + 1e-12)
        assert r.energy_kwh.sum() == pytest.approx(r.load_kw.sum() / 60)


@pytest.mark.parametrize("mode", ["equal_share", "proportional", "fifo"])
def test_load_never_exceeds_cap(mode):
    cfg = load_config(overrides={"site": {"lm_mode": mode, "grid_cap_kw": 300.0}})
    cat = real_catalog(cfg)
    sa = random_sessions(3000, 3 * 1440, seed=3)
    fleet = sample_fleet(len(sa), cat, cfg, np.random.default_rng(4))
    capped = simulate(sa, fleet, cat.power_kw, cfg)
    free = simulate(sa, fleet, cat.power_kw, cfg, grid_cap_kw=np.inf)
    assert free.load_kw.max() > 300.0
    assert capped.load_kw.max() <= 300.0 + 1e-9
    assert capped.binding_minutes > 0
    assert capped.energy_kwh.sum() < free.energy_kwh.sum()


def test_max_concurrency():
    sa = SessionArrays([0, 0, 5, 10], [10, 5, 15, 12], 150.0, n_minutes=20)
    assert sa.max_concurrency() == 2


def test_unsorted_sessions_rejected():
    with pytest.raises(ValueError, match="sorted"):
        SessionArrays([5, 0], [10, 10], 150.0, n_minutes=20)
