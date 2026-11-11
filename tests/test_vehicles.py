import numpy as np
import pytest

from chargepark.curves import N_SOC
from chargepark.vehicles import Catalog, sample_fleet


def test_catalog_normalizes_share():
    cat = Catalog.flat([50, 100, 150], [40, 60, 80], share=[1, 1, 2])
    assert cat.share == pytest.approx([0.25, 0.25, 0.5])
    assert cat.power_kw.shape == (3, N_SOC)


def test_catalog_rejects_bad_shapes():
    with pytest.raises(ValueError):
        Catalog(np.ones((2, 50)), [40, 60], [1, 1], ["a", "b"])
    with pytest.raises(ValueError):
        Catalog(np.ones((2, N_SOC)), [40, 60], [0, 0], ["a", "b"])


def test_sampling_is_reproducible(cfg):
    cat = Catalog.flat([50, 100], [40, 80])
    a = sample_fleet(100, cat, cfg, np.random.default_rng(1))
    b = sample_fleet(100, cat, cfg, np.random.default_rng(1))
    assert np.array_equal(a.vtype, b.vtype) and np.array_equal(a.soc0, b.soc0)


def test_sampled_values_follow_catalog(cfg):
    cat = Catalog.flat([50, 100, 150], [40, 60, 80], share=[0.2, 0.3, 0.5])
    f = sample_fleet(20_000, cat, cfg, np.random.default_rng(0))
    for k in range(3):
        m = f.vtype == k
        assert np.all(f.capacity_kwh[m] == cat.battery_kwh[k])
        assert abs(m.mean() - cat.share[k]) < 0.015
    a, b = cfg["vehicles"]["arrival_soc_beta"].values()
    assert abs(f.soc0.mean() - a / (a + b)) < 0.005
