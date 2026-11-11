from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chargepark.io import (alignment_shift, load_prices, load_sessions, parse_tz,
                          synthetic_prices, year_index_15)

DATA = Path(__file__).resolve().parents[1] / "data" / "sessions_dummy.csv"


def write(tmp_path, text, name="s.csv"):
    p = tmp_path / name
    p.write_text(text.strip() + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------- sessions
def test_simple_session(tmp_path, cfg):
    s, rep = load_sessions(write(tmp_path, "date,arrival_time,end_time\n2025-03-10,10:15,10:45"), cfg)
    assert len(s) == 1 and rep.n_dropped == 0
    assert s.loc[0, "start"] == pd.Timestamp("2025-03-10 10:15")
    assert s.loc[0, "end"] == pd.Timestamp("2025-03-10 10:45")
    assert s.loc[0, "duration_min"] == 30
    assert s.loc[0, "charger_kw"] == 150.0
    assert not s.loc[0, "wrapped"]


def test_midnight_crossing_session(tmp_path, cfg):
    s, rep = load_sessions(write(tmp_path, "date,arrival_time,end_time\n2025-03-10,23:50,00:20"), cfg)
    assert s.loc[0, "end"] == pd.Timestamp("2025-03-11 00:20")
    assert s.loc[0, "duration_min"] == 30
    assert s.loc[0, "wrapped"] and rep.n_wrapped == 1


def test_equal_times_wrap_to_24h(tmp_path, cfg):
    s, _ = load_sessions(write(tmp_path, "date,arrival_time,end_time\n2025-03-10,08:00,08:00"), cfg)
    assert s.loc[0, "duration_min"] == 1440


def test_year_end_crossing_is_kept_and_reported(tmp_path, cfg):
    s, rep = load_sessions(write(tmp_path, "date,arrival_time,end_time\n2025-12-31,23:50,00:30"), cfg)
    assert len(s) == 1 and rep.n_truncated == 1


def test_invalid_rows_dropped_with_reasons(tmp_path, cfg):
    text = """
date,arrival_time,end_time
2025-03-10,10:00,10:30
2025-13-01,10:00,10:30
2025-03-10,25:00,10:30
2025-03-10,10:00,
2025-03-10,10:00,1030
2024-03-10,10:00,10:30
2025-03-10,11:00,11:30
"""
    s, rep = load_sessions(write(tmp_path, text), cfg)
    assert rep.n_read == 7 and rep.n_kept == 2 and rep.n_dropped == 5
    assert rep.dropped == {
        "missing value": 1,
        "invalid date (expected YYYY-MM-DD)": 1,
        "invalid arrival_time (expected HH:MM)": 1,
        "invalid end_time (expected HH:MM)": 1,
        "date outside modeled year 2025": 1,
    }
    assert sorted(rep.dropped_rows["row"]) == [2, 3, 4, 5, 6]
    assert "dropped" in str(rep)


def test_multiple_years_require_config(tmp_path, cfg):
    p = write(tmp_path, "date,arrival_time,end_time\n2025-01-01,10:00,10:30\n2024-01-01,10:00,10:30")
    with pytest.raises(ValueError, match="sessions.year"):
        load_sessions(p, cfg)


def test_max_duration_filter(tmp_path, cfg):
    cfg["sessions"]["max_duration_min"] = 120
    p = write(tmp_path, "date,arrival_time,end_time\n2025-01-01,10:00,10:30\n2025-01-01,10:00,09:00")
    s, rep = load_sessions(p, cfg)
    assert len(s) == 1 and rep.n_dropped == 1


def test_missing_column_raises(tmp_path, cfg):
    with pytest.raises(ValueError, match="missing required columns"):
        load_sessions(write(tmp_path, "date,start\n2025-01-01,10:00"), cfg)


def test_sorted_by_start(tmp_path, cfg):
    p = write(tmp_path, "date,arrival_time,end_time\n2025-01-02,10:00,10:30\n2025-01-01,10:00,10:30")
    s, _ = load_sessions(p, cfg)
    assert s["start"].is_monotonic_increasing
    assert list(s["row"]) == [2, 1]


def test_dummy_file_loads(cfg):
    s, rep = load_sessions(DATA, cfg)
    assert rep.year == 2025 and rep.n_read == 11250
    assert rep.n_kept + rep.n_dropped == rep.n_read
    assert (s["duration_min"] > 0).all()
    assert rep.n_wrapped == 63




# ---------------------------------------------------------------- prices
DA_NC = DATA.parent / "prices" / "DA_{year}_15min_Germany_SMARD.nc"


def csv_prices(tmp_path, cfg, ts, da, idp, tz="local"):
    """Write one CSV holding both series and point prices.da / prices.id at it."""
    p = tmp_path / "p.csv"
    pd.DataFrame({"timestamp": ts, "da_eur_mwh": da, "id_eur_mwh": idp}).to_csv(p, index=False)
    for prod, col in (("da", "da_eur_mwh"), ("id", "id_eur_mwh")):
        cfg["prices"][prod] = {"path": str(p), "variable": col, "timestamp": "timestamp",
                               "timestamps_tz": tz}
    return p


def test_hourly_prices_forward_filled(tmp_path, cfg):
    hours = pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left")
    x = np.arange(len(hours), dtype=float)
    csv_prices(tmp_path, cfg, hours.strftime("%Y-%m-%d %H:%M"), x, x + 0.5)
    pr, rep = load_prices(cfg, 2025)
    assert pr.index.equals(year_index_15(2025))
    assert list(pr["da"].iloc[:5]) == [0, 0, 0, 0, 1]
    assert pr["id"].iloc[4] == 1.5
    assert rep.series["da"].max_fill_steps == 3


def test_utc_aware_timestamps_converted_to_local(tmp_path, cfg):
    # 2025-01-01 00:00 Berlin (CET, UTC+1) is 2024-12-31 23:00 UTC
    hours = pd.date_range("2024-12-31 23:00", "2025-12-31 23:00", freq="h", inclusive="left", tz="UTC")
    da = np.full(len(hours), 50.0)
    da[0] = 99.0
    csv_prices(tmp_path, cfg, hours.strftime("%Y-%m-%dT%H:%M:%S%z"), da, 60.0)
    pr, rep = load_prices(cfg, 2025)
    assert pr["da"].iloc[0] == 99.0                        # UTC 23:00 lands on local 00:00
    assert rep.series["da"].n_duplicate_timestamps == 1    # October fall-back hour averaged
    assert rep.series["da"].max_fill_steps == 7            # March spring-forward gap


def test_fixed_offset_cet_shifts_summer_only(tmp_path, cfg):
    # value = hour of day in CET all year; in local time (CEST) summer values
    # appear one hour later
    q = pd.date_range("2025-01-01", "2026-01-01", freq="15min", inclusive="left")
    x = q.hour.to_numpy().astype(float)
    csv_prices(tmp_path, cfg, q.strftime("%Y-%m-%d %H:%M"), x, x, tz="UTC+01:00")
    pr, rep = load_prices(cfg, 2025)
    assert pr.loc["2025-01-15 10:00", "id"] == 10.0        # winter: CET = local
    assert pr.loc["2025-07-15 10:00", "id"] == 9.0         # summer: 10:00 CEST = 09:00 CET
    assert rep.series["id"].n_duplicate_timestamps == 4    # 26 Oct, 02:00 local occurs twice


def test_price_gap_too_long_raises(tmp_path, cfg):
    hours = pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left")
    hours = hours[(hours < "2025-05-01") | (hours >= "2025-05-02")]
    csv_prices(tmp_path, cfg, hours.strftime("%Y-%m-%d %H:%M"), 1.0, 1.0)
    with pytest.raises(ValueError, match="price gap"):
        load_prices(cfg, 2025)


def test_alignment_check_flags_summer_shift(tmp_path, cfg):
    q = year_index_15(2025)
    rng = np.random.default_rng(0)
    da = np.sin(2 * np.pi * (q.hour + q.minute / 60) / 24) * 50 + rng.normal(0, 1, len(q))
    shifted = np.where(np.isin(q.month, [4, 5, 6, 7, 8, 9]), np.roll(da, 4), da)   # ID 1 h late in summer
    rep = alignment_shift(pd.DataFrame({"da": da, "id": shifted}, index=q))
    assert rep == {"winter": 0, "summer": -60}


def test_parse_tz():
    assert parse_tz("local") is None
    assert parse_tz("UTC+01:00").utcoffset(None).total_seconds() == 3600
    assert parse_tz("UTC-0530").utcoffset(None).total_seconds() == -19800
    assert parse_tz("Europe/Berlin") == "Europe/Berlin"


@pytest.mark.skipif(not Path(str(DA_NC).format(year=2025)).exists(), reason="DA 2025 file not present")
def test_provided_da_2025_loads(cfg):
    pr, rep = load_prices(cfg, 2025)
    assert list(pr.columns) == ["da"]                       # no ID configured
    assert pr.index.equals(year_index_15(2025)) and np.isfinite(pr.to_numpy()).all()
    r = rep.series["da"]
    assert r.n_read == 35040
    assert r.n_duplicate_timestamps == 4                    # October repeated hour, averaged
    assert r.max_fill_steps == 4                            # March gap filled from 01:45
    assert r.n_in_dst_gap == 0                              # local time, as the file states
    assert "no ID data" in rep.source()


@pytest.mark.skipif(not Path(str(DA_NC).format(year=2026)).exists(), reason="DA 2026 file not present")
def test_provided_da_2026_is_incomplete(cfg):
    with pytest.raises(ValueError, match="price gap"):
        load_prices(cfg, 2026)


def test_mode_needing_id_without_id_raises(tmp_path, cfg):
    hours = pd.date_range("2025-01-01", "2026-01-01", freq="h", inclusive="left")
    csv_prices(tmp_path, cfg, hours.strftime("%Y-%m-%d %H:%M"), 1.0, 1.0)
    cfg["prices"]["id"]["path"] = None
    cfg["procurement"]["mode"] = "forecast"
    with pytest.raises(ValueError, match="needs ID prices"):
        load_prices(cfg, 2025)
    cfg["procurement"]["mode"] = "da_only"
    pr, _ = load_prices(cfg, 2025)
    assert list(pr.columns) == ["da"]


def test_synthetic_prices_cover_year(cfg):
    pr, rep = synthetic_prices(2025, cfg)
    assert len(pr) == 365 * 96 and rep.synthetic
    assert np.isfinite(pr.to_numpy()).all()
    assert "SYNTHETIC" in str(rep)
