"""Small drawing helpers shared by the plot sections."""

from __future__ import annotations

import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .style import CAP_LINE, TEXT_MUTED


def heatmap(fig, ax, mat: pd.DataFrame, cmap, cbar_label: str, norm=None, vmin=None, vmax=None,
            hours: float = 24.0):
    """Day (y, top = 1 Jan) x time of day (x, hours) heatmap of a day_matrix."""
    d0 = mdates.date2num(mat.index[0])
    d1 = mdates.date2num(mat.index[-1] + pd.Timedelta(days=1))
    im = ax.imshow(np.ma.masked_invalid(mat.to_numpy(float)), aspect="auto", cmap=cmap, norm=norm,
                   vmin=None if norm else vmin, vmax=None if norm else vmax,
                   extent=(0, hours, d1, d0), interpolation="nearest", rasterized=True)
    ax.yaxis_date()
    ax.yaxis.set_major_locator(mdates.MonthLocator())
    ax.yaxis.set_major_formatter(mdates.DateFormatter("%b"))
    hour_axis(ax, hours)
    ax.grid(False)
    ax.set_ylabel("Day of year (date)")
    cb = fig.colorbar(im, ax=ax, pad=0.015, fraction=0.04)
    cb.set_label(cbar_label)
    cb.outline.set_visible(False)
    if norm is not None and getattr(norm, "vcenter", None) is not None:
        lo, c, hi = norm.vmin, norm.vcenter, norm.vmax
        cb.set_ticks(sorted(set(np.round(np.r_[np.linspace(lo, c, 3), np.linspace(c, hi, 5)]))))
    return im


def hour_axis(ax, hours: float = 24.0, step: int = 3, label: str = "Time of day (h)"):
    ax.set_xlim(0, hours)
    ax.set_xticks(np.arange(0, hours + 0.1, step))
    ax.set_xlabel(label)


def month_axis(ax, fmt: str = "%b"):
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter(fmt))


def day_axis(ax, fmt: str = "%d %b"):
    loc = mdates.AutoDateLocator(minticks=4, maxticks=10)
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))


def cap_line(ax, cap_kw: float, horizontal: bool = True):
    f = ax.axhline if horizontal else ax.axvline
    return f(cap_kw, color=CAP_LINE, lw=1.1, ls="--", label=f"Grid cap ({cap_kw:,.0f} kW)", zorder=5)


def percentile_lines(ax, values, color="#333333", label_prefix=""):
    """P10 / P50 / P90 vertical lines with labels for the legend."""
    p10, p50, p90 = np.percentile(values, [10, 50, 90])
    ax.axvline(p10, color=color, ls=":", lw=1.2, label=f"{label_prefix}P10 / P90")
    ax.axvline(p90, color=color, ls=":", lw=1.2)
    ax.axvline(p50, color=color, ls="-", lw=1.4, label=f"{label_prefix}P50")
    return p10, p50, p90


def hours_of_day(n: int) -> np.ndarray:
    """Slot start times in hours for n slots per day."""
    return np.arange(n) * 24.0 / n


def muted(ax, x, y, text, **kw):
    ax.text(x, y, text, color=TEXT_MUTED, fontsize=7, **kw)
