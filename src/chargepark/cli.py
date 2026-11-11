"""Command line entry point.

  python -m chargepark run --sessions data/sessions_dummy.csv
  python -m chargepark run --sessions data/sessions_dummy.csv --k all --runs 20
  python -m chargepark run --sessions data/sessions_dummy.csv --synthetic-prices
  python -m chargepark curves --elbow
  python -m chargepark compare-k --sessions data/sessions_dummy.csv --ks all,3,5,8,12

A call without subcommand (python -m chargepark --sessions ...) means `run`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .config import load_config
from .curves import load_curves
from .io import load_prices, load_sessions, synthetic_prices
from .montecarlo import compare_k, run_montecarlo
from .report import format_summary, plot_overview, summary_table, vehicle_info
from .representatives import build_representatives, elbow, plot_elbow, write_outputs
from .vehicles import Catalog

COMMANDS = ("run", "curves", "compare-k")


def parse_k(s: str):
    return "all" if s == "all" else int(s)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="chargepark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    def common(p, sessions=True):
        p.add_argument("--config", help="config YAML (default: config.yaml in the project root)")
        p.add_argument("--k", type=parse_k, help='representatives.k override (int or "all")')
        if sessions:
            p.add_argument("--sessions", required=True, help="sessions CSV")
            p.add_argument("--prices-da", help="DA price file (overrides prices.da.path)")
            p.add_argument("--prices-id", help="ID price file (overrides prices.id.path)")
            p.add_argument("--synthetic-prices", action="store_true",
                           help="use SYNTHETIC prices instead of the price files (pipeline testing only)")
            p.add_argument("--runs", type=int, help="Monte Carlo runs (overrides config)")
            p.add_argument("--out", help="output directory (overrides output.dir)")

    r = sub.add_parser("run", help="Monte Carlo cost run")
    common(r)
    r.add_argument("--no-plot", action="store_true")

    c = sub.add_parser("curves", help="build representative curves (and elbow analysis)")
    common(c, sessions=False)
    c.add_argument("--elbow", action="store_true", help="error metrics for k = 2..elbow_k_max")

    k = sub.add_parser("compare-k", help="compare simulation results across k")
    common(k)
    k.add_argument("--ks", default="all,3,5,8,12", help="comma-separated k values")
    return ap


def _sessions_and_prices(args, cfg):
    sessions, srep = load_sessions(args.sessions, cfg)
    print(srep)
    if srep.n_kept == 0:
        raise SystemExit("No valid sessions.")
    if args.synthetic_prices:
        if args.prices_da or args.prices_id:
            raise SystemExit("--synthetic-prices cannot be combined with price files.")
        prices, prep = synthetic_prices(srep.year, cfg)
    else:
        paths = {"da": args.prices_da, "id": args.prices_id}
        if not (paths["da"] or cfg["prices"]["da"]["path"]):
            print("No DA price file: set prices.da.path, pass --prices-da, "
                  "or use --synthetic-prices.", file=sys.stderr)
            raise SystemExit(2)
        prices, prep = load_prices(cfg, srep.year, paths)
    print(prep)
    return sessions, srep, prices, prep


def cmd_run(args, cfg) -> int:
    sessions, srep, prices, prep = _sessions_and_prices(args, cfg)
    rep = build_representatives(load_curves(cfg), cfg)
    vinfo = vehicle_info(rep, cfg)
    print(vinfo)

    t0 = time.time()
    res = run_montecarlo(sessions, srep.year, prices, cfg, Catalog.from_representatives(rep), args.runs)
    runtime = time.time() - t0

    out = Path(args.out or cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    res.runs.to_csv(out / "montecarlo_runs.csv", index=False)
    summary_table(res.runs).to_csv(out / "summary.csv")
    rep.table.to_csv(out / "vehicle_catalog.csv", index=False)
    if srep.n_dropped:
        srep.dropped_rows.to_csv(out / "dropped_sessions.csv", index=False)
    text = format_summary(res, cfg, srep, prep, runtime, vinfo)
    (out / "summary.txt").write_text(text + "\n", encoding="utf-8")
    print()
    print(text)
    if not args.no_plot:
        plot_overview(res, sessions, cfg, out / "overview.png", prep.synthetic)
    print(f"\nOutputs written to {out.resolve()}")
    return 0


def cmd_curves(args, cfg) -> int:
    cs = load_curves(cfg)
    rep = build_representatives(cs, cfg)
    print("\n".join(rep.log))
    cols = ["cluster", "medoid_id", "brand", "model", "battery_kwh", "peak_kw", "weight",
            "cluster_size", "mae_kw", "t10_80_mad_min"]
    print(rep.table[cols].round(3).to_string(index=False))
    for p in write_outputs(rep, cfg):
        print(f"wrote {p}")
    if args.elbow:
        df = elbow(cs, cfg)
        out = Path(cfg["representatives"]["output_dir"])
        df.to_csv(out / "elbow.csv", index=False)
        plot_elbow(df, cfg, out / "elbow.png")
        print(df.round(3).to_string(index=False))
        print(f"wrote {out / 'elbow.csv'} and {out / 'elbow.png'}")
    return 0


def cmd_compare_k(args, cfg) -> int:
    sessions, srep, prices, prep = _sessions_and_prices(args, cfg)
    ks = [parse_k(x.strip()) for x in args.ks.split(",") if x.strip()]
    t0 = time.time()
    df = compare_k(sessions, srep.year, prices, cfg, ks, args.runs)
    out = Path(args.out or cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "compare_k.csv", index=False)
    runs = args.runs or cfg["montecarlo"]["runs"]
    print(f"\nk comparison, {runs} runs each, P50 over runs ({time.time() - t0:.0f} s); "
          f"prices: {prep.source()}")
    show = ["k", "n_types", "curve_mae_kw", "t10_80_mad_min", "energy_mwh", "energy_dev_pct",
            "energy_lost_mwh", "energy_lost_dev_mwh", "eur_mwh", "eur_mwh_dev"]
    print(df[show].round(3).to_string(index=False))
    print(f"wrote {out / 'compare_k.csv'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] not in COMMANDS and argv[0] not in ("-h", "--help"):
        argv = ["run"] + argv
    args = build_parser().parse_args(argv)
    overrides = {"representatives": {"k": args.k}} if args.k is not None else None
    cfg = load_config(args.config, overrides)
    try:
        return {"run": cmd_run, "curves": cmd_curves, "compare-k": cmd_compare_k}[args.command](args, cfg)
    except SystemExit as e:
        if isinstance(e.code, str):
            print(e.code, file=sys.stderr)
            return 1
        return int(e.code or 0)


if __name__ == "__main__":
    sys.exit(main())
