#!/usr/bin/env python3
"""Plot OSU point-to-point latency & bandwidth for CPU (host) buffers.

Reads `osu_latency_H_<mode>.dat` and `osu_bw_H_<mode>.dat` from
`osu/results/` (produced by `osu/run_host.sh`) and draws:

  * left subplot  — latency (us) vs. message size
  * right subplot — bandwidth (GB/s) vs. message size

One line per CPU affinity mode (INTRA_CCD / INTER_CCD / INTER_SOCKET). Unlike
the GPU-buffer (`D D`) sweeps in `osu_ccd_plot.py`/`osu_xcd_plot.py`, there is
no SDMA/BLIT copy-engine axis here — that toggle only affects a GPU-side copy
engine, which is irrelevant when neither buffer ever touches the GPU.

Usage:
    python osu_host_plot.py
    python osu_host_plot.py --modes INTRA_CCD INTER_CCD
    python osu_host_plot.py --results-dir ../osu/results --output osu_host.png --show
"""
import argparse
from pathlib import Path

from common_style import MARKERS, SeriesStyler, create_figure, finalize, load_osu, plot_line, style_axis

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RESULTS_DIR = SCRIPT_DIR.parent / "osu" / "results"
DEFAULT_MODES = ["INTRA_CCD", "INTER_CCD", "INTER_SOCKET"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="Directory with osu_*_H_*.dat files")
    parser.add_argument("--modes", nargs="+", default=DEFAULT_MODES, help="CPU affinity modes to plot")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "plots" / "osu_host_plot.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    fig, ax_lat, ax_bw = create_figure("OSU Point-to-Point: CPU Affinity Sweep (CPU buffers, H H)")
    styler = SeriesStyler()

    for mode, marker in zip(args.modes, MARKERS):
        color = styler.color(mode)

        lat = load_osu(args.results_dir / f"osu_latency_H_{mode}.dat")
        if lat is None:
            print(f"  [skip] no latency data for {mode}")
        else:
            size, latency_us = lat
            plot_line(ax_lat, size, latency_us, color=color, marker=marker, label=mode)

        bw = load_osu(args.results_dir / f"osu_bw_H_{mode}.dat")
        if bw is None:
            print(f"  [skip] no bandwidth data for {mode}")
        else:
            size, bw_mbs = bw
            plot_line(ax_bw, size, bw_mbs / 1000.0, color=color, marker=marker, label=mode)

    style_axis(ax_lat, "Message Size (Bytes)", "Latency (us)", "Latency")
    style_axis(ax_bw, "Message Size (Bytes)", "Bandwidth (GB/s)", "Bandwidth")

    finalize(fig, ax_lat, ax_bw, args.output, show=args.show)


if __name__ == "__main__":
    main()
