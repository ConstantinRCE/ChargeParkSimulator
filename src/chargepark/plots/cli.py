"""Plot a results folder.

  python -m chargepark.plots --run results/<run_id>
  python -m chargepark.plots --run results/<run_id> --only 03          (or: load, 03_load; comma list ok)
  python -m chargepark.plots --run results/<run_id> --html             (needs plotly)

Writes <name>.pdf + <name>.png per plot and report.pdf to
results/plots/<run_id>/ (next to the run folder). Reads only the files of the
run folder; never re-runs the simulation.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt

from .catalog import TITLES, build_catalog, section_key
from .data import RunData
from .report import example_days_page, title_page
from .style import save, set_output_dir


def plot_dir_for(run_dir: Path) -> Path:
    """results/<run_id>/ -> results/plots/<run_id>/"""
    run_dir = Path(run_dir).resolve()
    return run_dir.parent / "plots" / run_dir.name


# --- worker side (one RunData per process) --------------------------------------
_W: dict = {}


def _init_worker(run_dir: str, out: str) -> None:
    _W["d"] = RunData(run_dir)
    _W["specs"] = build_catalog(_W["d"])
    set_output_dir(out)


def _render(i: int) -> tuple[int, float]:
    t = time.time()
    spec = _W["specs"][i]
    fig = spec.make(_W["d"])
    save(fig, spec.name)
    plt.close(fig)
    return i, time.time() - t


def _pages(d: RunData, out: Path) -> list[Path]:
    paths = []
    for name, make in (("_title", title_page), ("_example_days", example_days_page)):
        fig = make(d)
        paths.append(out / f"{name}.pdf")
        fig.savefig(paths[-1])
        plt.close(fig)
    return paths


def merge_pdfs(parts: list[Path], target: Path, title: str) -> Path:
    from pypdf import PdfWriter
    w = PdfWriter()
    for p in parts:
        w.append(str(p))
    w.add_metadata({"/Title": title})
    with open(target, "wb") as f:
        w.write(f)
    w.close()
    return target


def run_plots(run_dir: str | Path, only: list[str] | None = None, html: bool = False,
              report: bool = True, out_dir: str | Path | None = None, verbose: bool = True,
              workers: int | None = None) -> dict:
    """Render the catalog (optionally only some sections) to PDF + PNG and
    merge report.pdf (title page, example days, all plots in catalog order).
    workers: processes (default min(8, CPUs)); 1 renders in this process."""
    t0 = time.time()
    d = RunData(run_dir)
    out = set_output_dir(out_dir or plot_dir_for(d.run_dir))
    specs = build_catalog(d)
    idx = list(range(len(specs)))
    if only:
        keys = {section_key(s) for s in only}
        idx = [i for i in idx if specs[i].section in keys]
    workers = workers or min(8, os.cpu_count() or 1, max(len(idx), 1))
    times: dict[int, float] = {}
    if workers <= 1:
        _W.update(d=d, specs=specs)
        for i in idx:
            times[i] = _render(i)[1]
    else:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(d.run_dir), str(out))) as ex:
            for i, sec in ex.map(_render, idx):
                times[i] = sec
    if verbose:
        section = None
        for i in idx:
            if specs[i].section != section:
                section = specs[i].section
                print(f"{section} {TITLES[section]}")
            print(f"  {specs[i].name} ({times[i]:.1f} s)")
    written = [out / f"{specs[i].name}.{ext}" for i in idx for ext in ("pdf", "png")]
    if report:
        extra = _pages(d, out)
        merge_pdfs(extra + [out / f"{specs[i].name}.pdf" for i in idx], out / "report.pdf",
                   f"Charging park cost model, run {d.run_id}")
        for p in extra:
            p.unlink()
        written.append(out / "report.pdf")
    if html:
        from .html import write_html
        written.append(write_html(d, out / "explore.html"))
    pdfs = [p for p in written if p.suffix == ".pdf" and p.name != "report.pdf"]
    largest = max(pdfs, key=lambda p: p.stat().st_size) if pdfs else None
    res = {"out_dir": out, "files": written, "seconds": time.time() - t0, "largest_pdf": largest,
           "largest_pdf_kb": largest.stat().st_size / 1024 if largest else 0.0, "workers": workers}
    if verbose:
        print(f"{len(idx)} plots, {len(written)} files in {res['seconds']:.1f} s ({workers} workers) -> {out}")
        if largest:
            msg = f"largest plot PDF: {largest.name} ({res['largest_pdf_kb']:,.0f} kB)"
            if report:
                msg += f"; report.pdf {(out / 'report.pdf').stat().st_size / 1024:,.0f} kB"
            print(msg)
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m chargepark.plots", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="results/<run_id> folder")
    ap.add_argument("--only", help="section(s): 01..06 or inputs, mechanics, load, sessions, cost, montecarlo")
    ap.add_argument("--html", action="store_true", help="also write explore.html (plotly)")
    ap.add_argument("--no-report", action="store_true", help="skip the combined report.pdf")
    ap.add_argument("--out", help="output folder (default results/plots/<run_id>)")
    ap.add_argument("--workers", type=int, help="parallel processes (default min(8, CPUs); 1 = no pool)")
    a = ap.parse_args(argv)
    only = [s for s in a.only.split(",") if s.strip()] if a.only else None
    try:
        run_plots(a.run, only, a.html, not a.no_report, a.out, workers=a.workers)
    except (FileNotFoundError, ValueError) as e:
        print(e, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
