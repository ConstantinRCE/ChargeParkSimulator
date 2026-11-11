# EV DC charging-curve databank (Fastned)

Source: public brand pages of https://www.fastnedcharging.com/en/brands-overview,
collected 2026-11-10 from the structured data behind Fastned's charts (not digitised
from images). Check Fastned's terms of use before using the data in client
deliverables, and cite Fastned as the source.

## Contents
| File | Content |
|---|---|
| `raw/fastned_vehicles.csv` | Unmodified export, 647 versions incl. those without curve |
| `raw/fastned_charge_curves.csv` | Unmodified export, 26,742 measured points (1 % SOC steps) |
| `out/vehicles_databank.csv` | One row per version, metadata + derived fields (see below) |
| `out/curves_long.csv` | 312 versions x SOC 0..100: `power_kw`, `power_frac`, `point_type` |
| `out/curves_wide.csv` | Same curves, one row per `vehicle_id`, columns `soc_0`..`soc_100` (kW) |
| `out/qa_overview.png` | QA charts |
| `build_databank.py` | Rebuilds `out/` from `raw/` |

## Key fields in `vehicles_databank.csv`
- `charger_class_raw` / `charger_class_norm`: Fastned's charger-class label for the
  curve (50, 50+, 150+, 300, 300+, 400). Kept for later analysis; whether a curve is
  capped by that charger class is not confirmed.
- `battery_kwh_from_name`: parsed from the version name only. Empty for 11 versions
  with curves (mostly Tesla). Gross vs. net capacity is not specified by the source.
- `peak_kw_measured`, `soc_at_peak`, `avg_kw_10_80`: from the curve.
- `minutes_10_80`: 0.7 x capacity / avg_kw_10_80. Indicative only (assumes the name
  capacity is usable capacity; ignores charger losses).
- `curve_group_id`, `curve_shared_with_n`: versions with identical measured curves
  (150 distinct curves across 312 versions, mostly platform siblings).

## Processing rules for `curves_long` / `curves_wide`
| `point_type` | Rule |
|---|---|
| `measured` | Original Fastned value |
| `interpolated` | Linear, for 7 versions with small internal SOC gaps |
| `extrapolated_low` | Below first measured SOC (median 7 %): first value held constant |
| `extrapolated_high` | Above last measured SOC (median 92 %): linear ramp to 0 kW at 100 % |

## Known limitations
- Fastned states its curves reflect optimal battery conditions (about 30 °C).
  Cold-battery sessions charge slower; the curves overstate winter power.
- Curves are per version, not per vehicle in the field (no ageing, no preconditioning info).
- 335 of 647 versions have no curve (mostly CHAdeMO, AC-only, older models, some current ones).
