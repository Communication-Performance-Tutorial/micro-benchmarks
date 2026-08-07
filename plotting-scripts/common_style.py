"""Shared plot styling and .dat-file parsing helpers for the micro-benchmark plots.

Every script in this directory produces a figure with two side-by-side subplots
(latency on the left, bandwidth/busbw on the right) that share the same look:
grey dashed grid lines, log-log axes, square subplot aspect ratio, and a
consistent colour-per-series / linestyle-per-variant legend convention.

Import this module from the individual `*_plot.py` scripts; it is not meant to
be run directly.
"""
from __future__ import annotations

import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter

# ---------------------------------------------------------------------------
# Shared look & feel
# ---------------------------------------------------------------------------

FIGSIZE = (10, 4.5)
SUBPLOT_WSPACE = 0.18
GRID_KW = dict(which="both", linestyle="--", linewidth=0.6, color="0.75", alpha=0.9)

# Colour cycle used for the primary comparison axis (e.g. affinity mode,
# operation). Assigned in first-seen order so the same label always gets the
# same colour within one figure.
_COLOR_CYCLE = [
    "#1f77b4",  # blue
    "#d62728",  # red
    "#2ca02c",  # green
    "#9467bd",  # purple
    "#ff7f0e",  # orange
    "#17becf",  # teal
]

# Linestyle cycle used for the secondary comparison axis (e.g. copy engine,
# put vs. get).
LINESTYLES = ["-", "--", ":", "-."]

MARKERS = ["o", "s", "^", "D", "v", "P"]


def apply_global_style() -> None:
    """Set matplotlib rcParams shared by all plots in this repo."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "0.4",
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "font.size": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "xtick.color": "0.2",
            "ytick.color": "0.2",
        }
    )


class SeriesStyler:
    """Assigns a stable colour to each series label the first time it is seen."""

    def __init__(self) -> None:
        self._colors: dict[str, str] = {}
        self._palette = itertools.cycle(_COLOR_CYCLE)

    def color(self, series_label: str) -> str:
        if series_label not in self._colors:
            self._colors[series_label] = next(self._palette)
        return self._colors[series_label]


def create_figure(suptitle: str):
    """Create the standard latency (left) / bandwidth (right) figure."""
    apply_global_style()
    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=FIGSIZE, gridspec_kw={"wspace": SUBPLOT_WSPACE})
    fig.suptitle(suptitle, fontsize=13, fontweight="bold")
    return fig, ax_left, ax_right


def style_axis(
    ax,
    xlabel: str,
    ylabel: str,
    title: str,
    logx: bool = True,
    logy: bool = True,
    y_ticks: list[float] | None = None,
) -> None:
    if logx:
        ax.set_xscale("log", base=2)
    if logy:
        ax.set_yscale("log")
    if y_ticks is not None:
        set_plain_y_ticks(ax, y_ticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, **GRID_KW)
    for spine in ax.spines.values():
        spine.set_color("0.4")


def set_plain_y_ticks(ax, ticks: list[float]) -> None:
    """Force plain-number major y tick labels (e.g. 50, 60, 70) instead of the
    default log-scale power-of-ten notation (e.g. 5x10^1, 6x10^1, 7x10^1).

    Useful for log-scaled axes whose data spans less than one decade, where
    matplotlib's default LogLocator/LogFormatter produces scientific-looking
    labels even though a handful of plain ticks would read more naturally.
    """
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    ax.yaxis.set_minor_formatter(NullFormatter())


def plot_line(ax, x, y, *, color: str, linestyle: str = "-", marker: str = "o", label: str) -> None:
    ax.plot(
        x,
        y,
        color=color,
        linestyle=linestyle,
        marker=marker,
        markersize=4,
        linewidth=1.6,
        label=label,
    )


def finalize(fig, ax_left, ax_right, output_path: Path, show: bool = False) -> None:
    handles_l, labels_l = ax_left.get_legend_handles_labels()
    handles_r, labels_r = ax_right.get_legend_handles_labels()
    if labels_l:
        ax_left.legend(loc="best")
    if labels_r:
        ax_right.legend(loc="best")
    fig.subplots_adjust(left=0.08, right=0.96, top=0.86, bottom=0.12, wspace=SUBPLOT_WSPACE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    print(f"Saved plot to {output_path}")
    if show:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# .dat file parsers
# ---------------------------------------------------------------------------

def _data_lines(path: Path):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            yield line


def load_osu(path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Parse an `osu_latency`/`osu_bw` two-column .dat file: Size, Value."""
    if not path.is_file():
        return None
    sizes, values = [], []
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 2:
            continue
        sizes.append(float(parts[0]))
        values.append(float(parts[1]))
    if not sizes:
        return None
    return np.array(sizes), np.array(values)


def load_rccl(path: Path) -> dict | None:
    """Parse an rccl-tests (sendrecv_perf / all_reduce_perf) .dat file.

    Columns: size, count, type, redop, root,
             time_oop, algbw_oop, busbw_oop, wrong_oop,
             time_ip,  algbw_ip,  busbw_ip,  wrong_ip
    """
    if not path.is_file():
        return None
    sizes, time_oop, algbw_oop, busbw_oop, time_ip, algbw_ip, busbw_ip = ([] for _ in range(7))
    ngpus = None
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if line.startswith("#"):
                tokens = line.split()
                if "nGpus" in tokens:
                    idx = tokens.index("nGpus")
                    try:
                        ngpus = int(tokens[idx + 1])
                    except (IndexError, ValueError):
                        pass
                continue
            if not line:
                continue
            parts = line.split()
            if len(parts) < 12:
                continue
            try:
                size = float(parts[0])
            except ValueError:
                continue
            if size <= 0:
                continue  # log-log axes can't represent a 0 B message
            sizes.append(size)
            time_oop.append(float(parts[5]))
            algbw_oop.append(float(parts[6]))
            busbw_oop.append(float(parts[7]))
            time_ip.append(float(parts[9]))
            algbw_ip.append(float(parts[10]))
            busbw_ip.append(float(parts[11]))
    if not sizes:
        return None
    return {
        "size": np.array(sizes),
        "time_oop_us": np.array(time_oop),
        "algbw_oop_gbs": np.array(algbw_oop),
        "busbw_oop_gbs": np.array(busbw_oop),
        "time_ip_us": np.array(time_ip),
        "algbw_ip_gbs": np.array(algbw_ip),
        "busbw_ip_gbs": np.array(busbw_ip),
        "ngpus": ngpus,
    }


def load_rocshmem(path: Path) -> dict | None:
    """Parse a rocSHMEM `put_bw_*`/`get_bw_*` .dat file.

    Columns: Volume(B), Msg Size(B), #timed msgs, Latency(us), Bandwidth(GB/s), Msg Rate
    """
    if not path.is_file():
        return None
    sizes, latency, bandwidth = [], [], []
    for line in _data_lines(path):
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            size = float(parts[1])
        except ValueError:
            continue
        sizes.append(size)
        latency.append(float(parts[3]))
        bandwidth.append(float(parts[4]))
    if not sizes:
        return None
    return {
        "size": np.array(sizes),
        "latency_us": np.array(latency),
        "bandwidth_gbs": np.array(bandwidth),
    }
