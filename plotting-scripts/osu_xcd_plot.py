#!/usr/bin/env python3
"""Plot OSU point-to-point latency & bandwidth for the XCD (CPX-mode) sweep.

Reads `osu_latency_<mode>_<engine>.dat` and `osu_bw_<mode>_<engine>.dat` from
`osu/results/` (produced by `CPX_MODE=1 osu/run.sh`) and draws:

  * left subplot  — latency (us) vs. message size
  * right subplot — bandwidth (GB/s) vs. message size

By default this compares the two copy engines (SDMA vs. BLIT) for the
INTER_XCD case (same package, different XCD partition) — the axis CPX mode
actually varies, since no CPU pinning is applied in that mode.  Pass
`--modes INTER_XCD INTER_SOCKET` to instead compare the intra-package vs.
cross-package hop for a fixed engine.

Usage:
    python osu_xcd_plot.py
    python osu_xcd_plot.py --modes INTER_XCD INTER_SOCKET --engines SDMA
    python osu_xcd_plot.py --results-dir ../osu/results --output osu_xcd.png --show
"""
import argparse
from pathlib import Path

from common_style import LINESTYLES, MARKERS, SeriesStyler, create_figure, finalize, load_osu, plot_line, style_axis

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "osu" / "results"
DEFAULT_MODES = ["INTER_XCD"]
DEFAULT_ENGINES = ["SDMA", "BLIT"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory with osu_*.dat files")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="GPU affinity modes to plot (e.g. INTER_XCD, INTER_SOCKET)")
    parser.add_argument("--engines", nargs="+", default=DEFAULT_ENGINES, choices=["SDMA", "BLIT"], help="Copy engine(s) to plot")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "plots" / "osu_xcd_plot.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    fig, ax_lat, ax_bw = create_figure("OSU Point-to-Point: XCD Affinity Sweep (GPU buffers, D D, CPX mode)")
    styler = SeriesStyler()

    for mode in args.modes:
        color = styler.color(mode)
        for engine, linestyle, marker in zip(args.engines, LINESTYLES, MARKERS):
            label = mode if len(args.engines) == 1 else f"{mode} ({engine})"

            lat = load_osu(args.results_dir / f"osu_latency_{mode}_{engine}.dat")
            if lat is None:
                print(f"  [skip] no latency data for {mode} / {engine}")
            else:
                size, latency_us = lat
                plot_line(ax_lat, size, latency_us, color=color, linestyle=linestyle, marker=marker, label=label)

            bw = load_osu(args.results_dir / f"osu_bw_{mode}_{engine}.dat")
            if bw is None:
                print(f"  [skip] no bandwidth data for {mode} / {engine}")
            else:
                size, bw_mbs = bw
                plot_line(ax_bw, size, bw_mbs / 1000.0, color=color, linestyle=linestyle, marker=marker, label=label)

    style_axis(ax_lat, "Message Size (Bytes)", "Latency (us)", "Latency")
    style_axis(ax_bw, "Message Size (Bytes)", "Bandwidth (GB/s)", "Bandwidth")

    finalize(fig, ax_lat, ax_bw, args.output, show=args.show)


if __name__ == "__main__":
    main()
