"""Vehicle catalog (representative real curves with weights) and per-session
sampling of vehicle type and arrival SOC."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .curves import N_SOC


@dataclass
class Catalog:
    """Vehicle types available for sampling."""
    power_kw: np.ndarray      # (m, 101) DC kW at the connector, SOC 0..100 %
    battery_kwh: np.ndarray   # (m,)
    share: np.ndarray         # (m,) sampling probability, sums to 1
    names: list[str]

    def __post_init__(self):
        self.power_kw = np.ascontiguousarray(np.atleast_2d(self.power_kw), dtype=float)
        self.battery_kwh = np.atleast_1d(np.asarray(self.battery_kwh, dtype=float))
        share = np.atleast_1d(np.asarray(self.share, dtype=float))
        m = len(self.battery_kwh)
        if self.power_kw.shape != (m, N_SOC):
            raise ValueError(f"power_kw must have shape ({m}, {N_SOC}), got {self.power_kw.shape}")
        if share.shape != (m,) or share.sum() <= 0 or (share < 0).any():
            raise ValueError("share must be non-negative with a positive sum, one per vehicle type")
        self.share = share / share.sum()

    def __len__(self) -> int:
        return len(self.battery_kwh)

    @classmethod
    def from_representatives(cls, rep) -> "Catalog":
        power, battery, share, names = rep.catalog()
        return cls(power, battery, share, names)

    @classmethod
    def flat(cls, kw, battery_kwh, share=None) -> "Catalog":
        """Constant-power curves (tests, what-if studies)."""
        kw = np.atleast_1d(np.asarray(kw, float))
        bat = np.broadcast_to(np.asarray(battery_kwh, float), kw.shape)
        return cls(np.repeat(kw[:, None], N_SOC, axis=1), bat,
                   np.ones(len(kw)) if share is None else share, [f"flat_{x:g}kW" for x in kw])


def build_catalog(cfg: dict, k=None):
    """Load curves, reduce to representatives and return (Catalog, Representatives)."""
    from .curves import load_curves
    from .representatives import build_representatives
    rep = build_representatives(load_curves(cfg), cfg, k)
    return Catalog.from_representatives(rep), rep


@dataclass
class Fleet:
    """Per-session vehicle parameters for one Monte Carlo run."""
    vtype: np.ndarray        # int64, row of the catalog curve matrix
    capacity_kwh: np.ndarray
    soc0: np.ndarray         # arrival SOC in [0, 1]

    def __len__(self) -> int:
        return len(self.vtype)


def sample_fleet(n: int, catalog: Catalog, cfg: dict, rng: np.random.Generator) -> Fleet:
    vtype = rng.choice(len(catalog), size=n, p=catalog.share).astype(np.int64)
    b = cfg["vehicles"]["arrival_soc_beta"]
    soc0 = rng.beta(b["a"], b["b"], size=n)
    return Fleet(vtype=vtype, capacity_kwh=catalog.battery_kwh[vtype], soc0=soc0)


def fixed_fleet(catalog: Catalog, vtype, soc0) -> Fleet:
    """Fleet with explicit vehicle types and arrival SOCs (tests, scenarios)."""
    vtype = np.atleast_1d(np.asarray(vtype, np.int64))
    return Fleet(vtype=vtype, capacity_kwh=catalog.battery_kwh[vtype],
                 soc0=np.broadcast_to(np.asarray(soc0, float), vtype.shape).copy())
