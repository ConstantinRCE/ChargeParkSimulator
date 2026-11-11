import numpy as np
import pandas as pd
import pytest

from chargepark.io import year_index_15
from chargepark.procurement import price_energy, to_quarter_hours


def series(ts, values):
    return pd.DatetimeIndex(pd.to_datetime(ts)), np.asarray(values, float)


def test_to_quarter_hours():
    idx = pd.date_range("2025-01-01", periods=2, freq="15min")
    load = np.r_[np.full(15, 60.0), np.full(15, 120.0)]       # 60 kW then 120 kW
    e = to_quarter_hours(load, idx)
    assert list(e) == [15.0, 30.0]


def test_fixed_split_hand_case(cfg):
    cfg["procurement"].update(mode="fixed_split", share_ppa=0.3, share_da=0.6, share_id=0.1,
                              ppa_price_eur_mwh=70.0)
    idx = pd.date_range("2025-01-01", periods=2, freq="15min")
    e = pd.Series([1000.0, 2000.0], index=idx)                 # 1 and 2 MWh
    prices = pd.DataFrame({"da": [50.0, 100.0], "id": [80.0, 120.0]}, index=idx)
    r = price_energy(e, prices, cfg)
    assert r["cost_da_eur"] == pytest.approx(0.6 * (1 * 50 + 2 * 100))     # 150
    assert r["cost_id_eur"] == pytest.approx(0.1 * (1 * 80 + 2 * 120))     # 32
    assert r["cost_ppa_eur"] == pytest.approx(0.3 * 3 * 70)                # 63
    assert r["cost_eur"] == pytest.approx(245.0)


def test_forecast_hand_case(cfg):
    # two Mondays at 10:00 fall in the same profile bucket: residual 1 and 3 MWh,
    # forecast 2 MWh each -> ID volumes -1 and +1
    cfg["procurement"].update(mode="forecast", share_ppa=0.0)
    idx = pd.DatetimeIndex(["2025-01-06 10:00", "2025-01-13 10:00"])
    e = pd.Series([1000.0, 3000.0], index=idx)
    prices = pd.DataFrame({"da": [50.0, 50.0], "id": [40.0, 90.0]}, index=idx)
    r = price_energy(e, prices, cfg)
    assert r["cost_da_eur"] == pytest.approx(2 * 50 + 2 * 50)              # 200
    assert r["cost_id_eur"] == pytest.approx(-1 * 40 + 1 * 90)             # 50
    assert r["vol_id_net_mwh"] == pytest.approx(0.0)
    assert r["vol_id_abs_mwh"] == pytest.approx(2.0)


@pytest.mark.parametrize("mode", ["forecast", "fixed_split"])
def test_constant_prices_give_weighted_average(cfg, mode):
    # with DA = ID = 100 and PPA = 70 at 30 % PPA share, every MWh costs 91
    cfg["procurement"].update(mode=mode, share_ppa=0.3, share_da=0.6, share_id=0.1,
                              ppa_price_eur_mwh=70.0)
    idx = year_index_15(2025)
    rng = np.random.default_rng(0)
    e = pd.Series(rng.uniform(0, 100, len(idx)), index=idx)
    prices = pd.DataFrame({"da": 100.0, "id": 100.0}, index=idx)
    r = price_energy(e, prices, cfg)
    assert r["cost_eur"] / (e.sum() / 1000) == pytest.approx(91.0)


def test_index_mismatch_rejected(cfg):
    idx = pd.date_range("2025-01-01", periods=2, freq="15min")
    with pytest.raises(ValueError):
        price_energy(pd.Series([1.0, 2.0], index=idx),
                     pd.DataFrame({"da": [1.0, 1.0], "id": [1.0, 1.0]}, index=idx + pd.Timedelta("15min")), cfg)


def test_da_only_hand_case(cfg):
    cfg["procurement"].update(mode="da_only", share_ppa=0.3, ppa_price_eur_mwh=70.0)
    idx = pd.date_range("2025-01-01", periods=2, freq="15min")
    e = pd.Series([1000.0, 2000.0], index=idx)
    prices = pd.DataFrame({"da": [50.0, 100.0]}, index=idx)          # no ID column needed
    r = price_energy(e, prices, cfg)
    assert r["cost_da_eur"] == pytest.approx(0.7 * (1 * 50 + 2 * 100))     # 175
    assert r["cost_id_eur"] == 0.0
    assert r["cost_eur"] == pytest.approx(175 + 0.3 * 3 * 70)              # 238


def test_forecast_without_id_column_raises(cfg):
    cfg["procurement"]["mode"] = "forecast"
    idx = pd.date_range("2025-01-01", periods=2, freq="15min")
    with pytest.raises(ValueError, match="ID"):
        price_energy(pd.Series([1.0, 2.0], index=idx), pd.DataFrame({"da": [1.0, 1.0]}, index=idx), cfg)
