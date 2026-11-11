"""Load and validate config.yaml."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

LM_MODES = {"equal_share": 0, "proportional": 1, "fifo": 2}
PROCUREMENT_MODES = ("da_only", "forecast", "fixed_split")
MODES_NEEDING_ID = ("forecast", "fixed_split")

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.yaml"


class ConfigError(ValueError):
    pass


def deep_merge(base: dict, override: dict) -> dict:
    """Return a copy of base with override merged in (dicts merged recursively,
    everything else replaced)."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> dict[str, Any]:
    path = Path(path) if path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    validate_config(cfg)
    # relative data paths are relative to the config file's directory
    base = path.resolve().parent

    def absolute(p):
        return p if (p is None or Path(p).is_absolute()) else str(base / p)

    for prod in ("da", "id"):
        cfg["prices"][prod]["path"] = absolute(cfg["prices"][prod]["path"])
    for src in ("fastned", "evkx"):
        cfg["curves"][src]["dir"] = absolute(cfg["curves"][src]["dir"])
    r = cfg["representatives"]
    r["output_dir"] = absolute(r["output_dir"])
    r["weights_csv"] = absolute(r["weights_csv"])
    cfg["output"]["dir"] = absolute(cfg["output"]["dir"])
    return cfg


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


def validate_config(cfg: dict) -> None:
    site = cfg["site"]
    _require(site["charger_kw"] > 0, "site.charger_kw must be > 0")
    _require(site["grid_cap_kw"] > 0, "site.grid_cap_kw must be > 0")
    _require(0 < site["charger_efficiency"] <= 1, "site.charger_efficiency must be in (0, 1]")
    _require(site["lm_mode"] in LM_MODES, f"site.lm_mode must be one of {list(LM_MODES)}")

    s = cfg["sessions"]
    _require(s["max_duration_min"] is None or s["max_duration_min"] > 0,
             "sessions.max_duration_min must be null or > 0")

    v = cfg["vehicles"]
    _require(0 < v["target_soc"] <= 1, "vehicles.target_soc must be in (0, 1]")
    _require(v["arrival_soc_beta"]["a"] > 0 and v["arrival_soc_beta"]["b"] > 0,
             "vehicles.arrival_soc_beta a and b must be > 0")
    _require(0 < v["vehicle_efficiency"] <= 1, "vehicles.vehicle_efficiency must be in (0, 1]")
    td = v["temperature_derating"]
    _require(len(td["monthly_factor"]) == 12, "temperature_derating.monthly_factor needs 12 values")
    _require(all(0 < x <= 1.5 for x in td["monthly_factor"]),
             "temperature_derating.monthly_factor values must be in (0, 1.5]")

    p = cfg["procurement"]
    _require(p["mode"] in PROCUREMENT_MODES, f"procurement.mode must be one of {PROCUREMENT_MODES}")
    _require(0 <= p["share_ppa"] <= 1, "procurement.share_ppa must be in [0, 1]")
    if p["mode"] == "fixed_split":
        tot = p["share_ppa"] + p["share_da"] + p["share_id"]
        _require(abs(tot - 1) < 1e-9, f"fixed_split shares must sum to 1 (got {tot})")

    pr = cfg["prices"]
    _require(pr["max_fill_steps"] >= 0, "prices.max_fill_steps must be >= 0")
    for prod in ("da", "id"):
        _require(isinstance(pr.get(prod), dict), f"prices.{prod} must be a mapping")
        for key in ("path", "variable", "timestamp", "timestamps_tz"):
            _require(key in pr[prod], f"prices.{prod}.{key} is missing")

    r = cfg["representatives"]
    _require(r["k"] == "all" or (isinstance(r["k"], int) and r["k"] >= 1),
             'representatives.k must be a positive int or "all"')
    _require(r["battery_weight"] >= 0, "representatives.battery_weight must be >= 0")
    _require(isinstance(r["feature_soc_step"], int) and 100 % r["feature_soc_step"] == 0,
             "representatives.feature_soc_step must divide 100")
    _require(int(r["n_init"]) >= 1, "representatives.n_init must be >= 1")
    _require(r["default_weight"] >= 0, "representatives.default_weight must be >= 0")
    _require(len(cfg["curves"]["fastned"]["categories"]) > 0, "curves.fastned.categories is empty")

    mc = cfg["montecarlo"]
    _require(int(mc["runs"]) >= 1, "montecarlo.runs must be >= 1")
