from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from chargepark.curves import N_SOC, SOC_COLS, load_curves, load_evkx, load_fastned

FASTNED = Path(__file__).resolve().parents[1] / "data" / "ev_curves" / "fastned" / "out"
needs_fastned = pytest.mark.skipif(not (FASTNED / "curves_wide.csv").exists(),
                                   reason="Fastned databank not extracted")


# ---------------------------------------------------------------- Fastned (real data)
@needs_fastned
def test_fastned_filters_and_schema(cfg):
    cs = load_fastned(cfg)
    m = cs.meta
    assert cs.power_kw.shape == (len(m), N_SOC)
    assert (m["source"] == "fastned").all() and (m["category"] == "CAR").all()
    assert m["battery_kwh"].notna().all() and m["curve_id"].is_unique
    assert np.allclose(cs.power_kw.max(axis=1), m["peak_kw"])

    v = pd.read_csv(FASTNED / "vehicles_databank.csv")
    elig = v[(v.has_curve == 1) & v.battery_kwh_from_name.notna() & (v.category == "CAR")]
    assert len(m) == elig.curve_group_id.nunique()                 # one per group
    assert m["n_versions"].sum() == len(elig)                     # every version accounted for
    assert len(cs.versions) == len(elig)


@needs_fastned
def test_fastned_curve_values_match_databank(cfg):
    cs = load_fastned(cfg)
    w = pd.read_csv(FASTNED / "curves_wide.csv").set_index("vehicle_id")
    vid = int(cs.meta.loc[0, "curve_id"].split(":")[1])
    assert np.array_equal(cs.power_kw[0], w.loc[vid, SOC_COLS].to_numpy(float))


@needs_fastned
def test_fastned_categories_configurable(cfg):
    cfg["curves"]["fastned"]["categories"] = ["CAR", "VAN"]
    assert set(load_fastned(cfg).meta["category"]) == {"CAR", "VAN"}


# ---------------------------------------------------------------- evkx (fixture; real layout unverified)
def write_evkx(d: Path):
    (d / "out").mkdir(parents=True)
    rows = [
        # id, brand, model, quality, estimated, peak, cap
        (1, "Tesla", "Model 3", "ok", "FALSE", 170, 57.5),     # same brand/model as Fastned? depends on data
        (2, "Aiways", "U5", "ok", "FALSE", 90, 63.0),
        (3, "BYD", "Dolphin", "ok", "TRUE", 88, 60.4),          # estimated
        (4, "Fisker", "Ocean", "bad", "FALSE", 150, 106.0),     # quality
        (5, "Cupra", "Born", "ok", "FALSE", 900, 77.0),         # peak > 600
        (6, "Citroen", "e-C4", "ok", "FALSE", 100, None),       # no capacity
    ]
    v = pd.DataFrame(rows, columns=["vehicle_id", "brand", "model", "curve_quality", "estimated",
                                    "peak_kw", "usable_kwh_0_100"])
    v["version"] = ""
    v.to_csv(d / "out" / "vehicles.csv", index=False)
    curves = pd.DataFrame(np.tile(np.linspace(100, 10, N_SOC), (len(v), 1)), columns=SOC_COLS)
    curves.insert(0, "vehicle_id", v["vehicle_id"])
    curves.to_csv(d / "out" / "curves_wide.csv", index=False)


def test_evkx_filters(tmp_path, cfg):
    write_evkx(tmp_path)
    cfg["curves"]["evkx"].update(enabled=True, dir=str(tmp_path))
    cs = load_evkx(cfg)
    assert set(cs.meta["model"]) == {"Model 3", "U5"}
    assert (cs.meta["source"] == "evkx_measured").all()
    assert cs.meta.set_index("model").loc["U5", "battery_kwh"] == 63.0

    cfg["curves"]["evkx"]["include_estimated"] = True
    cs = load_evkx(cfg)
    assert cs.meta.set_index("model").loc["Dolphin", "source"] == "evkx_estimated"


@needs_fastned
def test_fastned_wins_over_evkx(tmp_path, cfg):
    write_evkx(tmp_path)
    cfg["curves"]["evkx"].update(enabled=True, dir=str(tmp_path))
    fast = load_fastned(cfg)
    both = load_curves(cfg)
    ev = both.meta[both.meta["source"] != "fastned"]
    fast_models = set(fast.versions["brand"].str.lower() + "|" + fast.versions["model"].str.lower())
    assert not any(f"{b.lower()}|{m.lower()}" in fast_models for b, m in zip(ev["brand"], ev["model"]))
    assert len(both) == len(fast) + len(ev)


def test_evkx_missing_dir_raises(tmp_path, cfg):
    cfg["curves"]["evkx"].update(enabled=True, dir=str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError):
        load_evkx(cfg)
