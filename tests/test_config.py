import pytest

from chargepark.config import ConfigError, load_config


def test_default_config_loads(cfg):
    assert cfg["site"]["grid_cap_kw"] == 500.0
    assert cfg["vehicles"]["target_soc"] == 0.80


def test_overrides_merge_without_losing_siblings():
    cfg = load_config(overrides={"site": {"grid_cap_kw": 300.0}})
    assert cfg["site"]["grid_cap_kw"] == 300.0
    assert cfg["site"]["charger_kw"] == 150.0


@pytest.mark.parametrize("override", [
    {"site": {"lm_mode": "round_robin"}},
    {"site": {"charger_efficiency": 1.2}},
    {"vehicles": {"target_soc": 0.0}},
    {"procurement": {"mode": "fixed_split", "share_ppa": 0.5, "share_da": 0.5, "share_id": 0.5}},
    {"procurement": {"mode": "spot"}},
])
def test_invalid_config_rejected(override):
    with pytest.raises(ConfigError):
        load_config(overrides=override)


def test_derating_needs_12_months():
    with pytest.raises(ConfigError):
        load_config(overrides={"vehicles": {"temperature_derating": {"enabled": True, "monthly_factor": [1.0]}}})


@pytest.mark.parametrize("k", [0, "some", -3])
def test_invalid_k_rejected(k):
    with pytest.raises(ConfigError):
        load_config(overrides={"representatives": {"k": k}})
