#!/usr/bin/env python3
"""Plot RCCL sendrecv_perf latency & bandwidth from rccl-tests/results/.

Reads `sendrecv_<label>.dat` files (produced by `rccl-tests/run.sh`, e.g.
`sendrecv_INTER_SOCKET.dat` from SPX mode and `sendrecv_cpx_INTER_XCD.dat`
from `CPX_MODE=1 run.sh`) and draws:

  * left subplot  — out-of-place time (us) vs. message size
  * right subplot — out-of-place algorithmic bandwidth (GB/s) vs. message size

One line per result file found; missing files are skipped with a warning so
this works whether you've only run SPX mode, only CPX mode, or both.

Usage:
    python rccl_sendrecv_plot.py
    python rccl_sendrecv_plot.py --results-dir ../rccl-tests/results --output sendrecv.png --show
"""
import argparse
from pathlib import Path

from common_style import MARKERS, SeriesStyler, create_figure, finalize, load_rccl, plot_line, style_axis

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "rccl-tests" / "results"

# (legend label, filename)
DEFAULT_FILES = [
    ("INTER_SOCKET", "sendrecv_INTER_SOCKET.dat"),
    ("INTER_XCD (CPX)", "sendrecv_cpx_INTER_XCD.dat"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory with sendrecv*.dat files")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "plots" / "rccl_sendrecv_plot.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    fig, ax_lat, ax_bw = create_figure("RCCL sendrecv_perf: Latency & Bandwidth")
    styler = SeriesStyler()

    for (label, filename), marker in zip(DEFAULT_FILES, MARKERS):
        data = load_rccl(args.results_dir / filename)
        if data is None:
            print(f"  [skip] no data for {label} ({filename})")
            continue
        color = styler.color(label)
        plot_line(ax_lat, data["size"], data["time_oop_us"], color=color, marker=marker, label=label)
        plot_line(ax_bw, data["size"], data["algbw_oop_gbs"], color=color, marker=marker, label=label)

    style_axis(ax_lat, "Message Size (Bytes)", "Time (us)", "Latency")
    style_axis(ax_bw, "Message Size (Bytes)", "Bandwidth (GB/s)", "Bandwidth (algbw)")

    finalize(fig, ax_lat, ax_bw, args.output, show=args.show)


if __name__ == "__main__":
    main()
