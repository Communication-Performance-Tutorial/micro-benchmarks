#!/usr/bin/env python3
"""Plot RCCL all_reduce_perf latency & busbw from rccl-tests/results/.

Reads every `all_reduce*.dat` file (produced by `rccl-tests/run.sh`) and draws:

  * left subplot  — out-of-place time (us) vs. message size
  * right subplot — out-of-place bus bandwidth (GB/s) vs. message size

Bus bandwidth (not algbw) is used on the right since that is the standard
RCCL metric for comparing collective performance against the theoretical
hardware peak, regardless of GPU count. Each matching file becomes one line,
labeled with the GPU count parsed from its header (falls back to the
filename if the header can't be parsed) — handy if you have runs from
different `-g`/node-size configurations sitting side by side.

Usage:
    python rccl_allreduce_plot.py
    python rccl_allreduce_plot.py --results-dir ../rccl-tests/results --output allreduce.png --show
"""
import argparse
from pathlib import Path

from common_style import MARKERS, SeriesStyler, create_figure, finalize, load_rccl, plot_line, style_axis

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "rccl-tests" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory with all_reduce*.dat files")
    parser.add_argument("--glob", default="all_reduce*.dat", help="Filename pattern to search for")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "plots" / "rccl_allreduce_plot.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    fig, ax_lat, ax_bw = create_figure("RCCL all_reduce_perf: Latency & Bus Bandwidth")
    styler = SeriesStyler()

    files = sorted(args.results_dir.glob(args.glob))
    if not files:
        print(f"  [skip] no files matching '{args.glob}' in {args.results_dir}")

    for path, marker in zip(files, MARKERS):
        data = load_rccl(path)
        if data is None:
            print(f"  [skip] could not parse {path.name}")
            continue
        label = f"{data['ngpus']} GPUs" if data["ngpus"] else path.stem
        color = styler.color(label)
        plot_line(ax_lat, data["size"], data["time_oop_us"], color=color, marker=marker, label=label)
        plot_line(ax_bw, data["size"], data["busbw_oop_gbs"], color=color, marker=marker, label=label)

    style_axis(ax_lat, "Message Size (Bytes)", "Time (us)", "Latency")
    style_axis(ax_bw, "Message Size (Bytes)", "Bus Bandwidth (GB/s)", "Bus Bandwidth", y_ticks=[50, 60, 70])

    finalize(fig, ax_lat, ax_bw, args.output, show=args.show)


if __name__ == "__main__":
    main()
