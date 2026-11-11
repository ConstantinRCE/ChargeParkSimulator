"""Shared plot style: colors, fonts, colormaps, figure footnotes and save()."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

GREEN, BLUE, NAVY, PURPLE, PINK, BLACK = "#005B4C", "#0000FF", "#00002C", "#6544FE", "#FF7F69", "#000000"
COLORS = {"green": GREEN, "blue": BLUE, "navy": NAVY, "purple": PURPLE, "pink": PINK, "black": BLACK}
# categorical order: always assigned in this order, never cycled
SERIES = [GREEN, BLUE, PURPLE, PINK, NAVY, BLACK]
GRID = "#E6E6E6"
MISSING = "#C8C8C8"          # heatmap cells that do not exist (DST gap)
TEXT_MUTED = "#555555"

# fixed roles
WITH_CAP, WITHOUT_CAP, CAP_LINE = GREEN, PURPLE, BLACK

PNG_DPI = 200
_OUT_DIR: Path | None = None


def tint(color: str, f: float) -> str:
    """Mix color with white: f = 0 gives the color, f = 1 white."""
    r, g, b = mcolors.to_rgb(color)
    return mcolors.to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def seq_cmap(color: str = GREEN, name: str | None = None) -> mcolors.Colormap:
    """Sequential: one hue, white to color. Missing cells in light grey."""
    cm = mcolors.LinearSegmentedColormap.from_list(name or f"seq_{color}", ["#FFFFFF", tint(color, 0.55), color])
    return cm.with_extremes(bad=MISSING)


def div_cmap() -> mcolors.Colormap:
    """Diverging: blue (negative), neutral grey midpoint, pink to navy (positive)."""
    cm = mcolors.LinearSegmentedColormap.from_list(
        "div_price", [BLUE, tint(BLUE, 0.6), "#F2F2F2", tint(PINK, 0.4), PINK, NAVY])
    return cm.with_extremes(bad=MISSING)


def apply_style() -> None:
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Liberation Sans", "Helvetica", "DejaVu Sans"],
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "legend.fontsize": 8, "legend.frameon": False,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
        "axes.edgecolor": "#666666", "xtick.color": "#444444", "ytick.color": "#444444",
        "axes.prop_cycle": matplotlib.cycler(color=SERIES),
        "figure.dpi": 100, "savefig.dpi": PNG_DPI, "pdf.fonttype": 3,          # Type 3 subsets: 3x faster to write than 42
        "lines.linewidth": 1.4, "patch.linewidth": 0,
    })


def figure(nrows=1, ncols=1, size=(10, 5), **kw):
    apply_style()
    return plt.subplots(nrows, ncols, figsize=size, **kw)


def footnote(fig, *lines: str) -> None:
    """Small notes at the bottom left (assumptions, data caveats)."""
    lines = [x for x in lines if x]
    if lines:
        fig.text(0.01, 0.005, "\n".join(lines), fontsize=7, color=TEXT_MUTED, ha="left", va="bottom")


def finish(fig, notes: tuple[str, ...] | list[str] = (), bottom: float | None = None) -> None:
    footnote(fig, *notes)
    n = len([x for x in notes if x])
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        fig.tight_layout(rect=(0, bottom if bottom is not None else 0.025 + 0.022 * n, 1, 1))


def tight(fig, rect=(0, 0, 1, 1)) -> None:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*tight_layout.*")
        fig.tight_layout(rect=rect)


def set_output_dir(path: str | Path) -> Path:
    global _OUT_DIR
    _OUT_DIR = Path(path)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    return _OUT_DIR


def save(fig, name: str, out_dir: str | Path | None = None) -> list[Path]:
    """Write <name>.pdf and <name>.png (200 dpi) to out_dir (default: the
    directory set by set_output_dir, i.e. results/plots/<run_id>/)."""
    d = Path(out_dir) if out_dir is not None else _OUT_DIR
    if d is None:
        raise RuntimeError("no output directory: call set_output_dir() or pass out_dir")
    d.mkdir(parents=True, exist_ok=True)
    paths = [d / f"{name}.pdf", d / f"{name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=PNG_DPI)
    return paths
