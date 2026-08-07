#!/usr/bin/env python3
"""Plot rocSHMEM put/get latency & bandwidth from rocshmem/results/.

Reads `put_bw_<mode>.dat` and `get_bw_<mode>.dat` (produced by
`rocshmem/run.sh`) and draws:

  * left subplot  — latency (us) vs. message size, Put and Get
  * right subplot — bandwidth (GB/s) vs. message size, Put and Get

One line per (mode, operation) combination: colour encodes the CPU affinity
mode (INTRA_CCD / INTER_CCD / INTER_SOCKET), linestyle encodes the operation
(Put = solid, Get = dashed).

Usage:
    python rocshmem_plot.py
    python rocshmem_plot.py --modes INTER_SOCKET
    python rocshmem_plot.py --results-dir ../rocshmem/results --output rocshmem.png --show
"""
import argparse
from pathlib import Path

from common_style import MARKERS, SeriesStyler, create_figure, finalize, load_rocshmem, plot_line, style_axis

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "rocshmem" / "results"
DEFAULT_MODES = ["INTRA_CCD", "INTER_CCD", "INTER_SOCKET"]

# operation label -> (file prefix, linestyle)
OPERATIONS = [("Put", "put_bw", "-"), ("Get", "get_bw", "--")]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory with put_bw_*.dat / get_bw_*.dat files")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="CPU affinity modes to plot")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "plots" / "rocshmem_plot.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    fig, ax_lat, ax_bw = create_figure("rocSHMEM Put/Get: Latency & Bandwidth (IPC backend)")
    styler = SeriesStyler()

    for mode in args.modes:
        color = styler.color(mode)
        for op_idx, (op_label, prefix, linestyle) in enumerate(OPERATIONS):
            marker = MARKERS[op_idx % len(MARKERS)]
            label = f"{mode} ({op_label})"

            data = load_rocshmem(args.results_dir / f"{prefix}_{mode}.dat")
            if data is None:
                print(f"  [skip] no data for {mode} / {op_label}")
                continue

            plot_line(ax_lat, data["size"], data["latency_us"], color=color, linestyle=linestyle, marker=marker, label=label)
            plot_line(ax_bw, data["size"], data["bandwidth_gbs"], color=color, linestyle=linestyle, marker=marker, label=label)

    style_axis(ax_lat, "Message Size (Bytes)", "Latency (us)", "Latency")
    style_axis(ax_bw, "Message Size (Bytes)", "Bandwidth (GB/s)", "Bandwidth")

    finalize(fig, ax_lat, ax_bw, args.output, show=args.show)


if __name__ == "__main__":
    main()
