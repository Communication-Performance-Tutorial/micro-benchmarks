#!/usr/bin/env python3
"""Plot OSU + RCCL + rocSHMEM latency & bandwidth together, INTER_SOCKET only.

`INTER_SOCKET` (cross-package, over xGMI/Infinity Fabric) is the one affinity
mode all three suites can produce, so it's used here to give a single,
clean, cross-suite comparison — one line per suite (colour), latency on the
left and bandwidth on the right:

  * OSU     — `osu/results/osu_latency_INTER_SOCKET_<engine>.dat` / `osu_bw_INTER_SOCKET_<engine>.dat`
  * RCCL    — `rccl-tests/results/sendrecv_INTER_SOCKET.dat` (out-of-place time / algbw)
  * rocSHMEM— `rocshmem/results/<op>_bw_INTER_SOCKET.dat` (Put by default)

Kept to three lines total (one per suite) so the comparison stays readable —
pass `--rocshmem-op both` if you also want rocSHMEM's Get line.

Usage:
    python combined_inter_socket_plot.py
    python combined_inter_socket_plot.py --rocshmem-op both --osu-engine BLIT
    python combined_inter_socket_plot.py --output combined.png --show
"""
import argparse
from pathlib import Path

from common_style import (
    MARKERS,
    SeriesStyler,
    create_figure,
    finalize,
    load_osu,
    load_rccl,
    load_rocshmem,
    plot_line,
    style_axis,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OSU_DIR = SCRIPT_DIR.parent / "osu" / "results"
DEFAULT_RCCL_DIR = SCRIPT_DIR.parent / "rccl-tests" / "results"
DEFAULT_ROCSHMEM_DIR = SCRIPT_DIR.parent / "rocshmem" / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--osu-results-dir", type=Path, default=DEFAULT_OSU_DIR)
    parser.add_argument("--rccl-results-dir", type=Path, default=DEFAULT_RCCL_DIR)
    parser.add_argument("--rocshmem-results-dir", type=Path, default=DEFAULT_ROCSHMEM_DIR)
    parser.add_argument("--osu-engine", default="SDMA", choices=["SDMA", "BLIT"], help="OSU copy engine to plot")
    parser.add_argument("--rocshmem-op", default="put", choices=["put", "get", "both"], help="rocSHMEM operation(s) to plot")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "plots" / "combined_inter_socket_plot.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive window")
    args = parser.parse_args()

    fig, ax_lat, ax_bw = create_figure("OSU vs. RCCL vs. rocSHMEM: Latency & Bandwidth (INTER_SOCKET)")
    styler = SeriesStyler()
    markers = iter(MARKERS)

    # --- OSU ---------------------------------------------------------------
    label = "OSU"
    color = styler.color(label)
    marker = next(markers)
    lat = load_osu(args.osu_results_dir / f"osu_latency_INTER_SOCKET_{args.osu_engine}.dat")
    if lat is None:
        print(f"  [skip] no OSU latency data ({args.osu_engine})")
    else:
        size, latency_us = lat
        plot_line(ax_lat, size, latency_us, color=color, marker=marker, label=label)
    bw = load_osu(args.osu_results_dir / f"osu_bw_INTER_SOCKET_{args.osu_engine}.dat")
    if bw is None:
        print(f"  [skip] no OSU bandwidth data ({args.osu_engine})")
    else:
        size, bw_mbs = bw
        plot_line(ax_bw, size, bw_mbs / 1000.0, color=color, marker=marker, label=label)

    # --- RCCL ----------------------------------------------------------------
    label = "RCCL"
    color = styler.color(label)
    marker = next(markers)
    rccl = load_rccl(args.rccl_results_dir / "sendrecv_INTER_SOCKET.dat")
    if rccl is None:
        print("  [skip] no RCCL sendrecv data")
    else:
        plot_line(ax_lat, rccl["size"], rccl["time_oop_us"], color=color, marker=marker, label=label)
        plot_line(ax_bw, rccl["size"], rccl["algbw_oop_gbs"], color=color, marker=marker, label=label)

    # --- rocSHMEM ------------------------------------------------------------
    ops = ["put", "get"] if args.rocshmem_op == "both" else [args.rocshmem_op]
    for op in ops:
        label = "rocSHMEM" if len(ops) == 1 else f"rocSHMEM ({op.capitalize()})"
        color = styler.color(label)
        marker = next(markers)
        data = load_rocshmem(args.rocshmem_results_dir / f"{op}_bw_INTER_SOCKET.dat")
        if data is None:
            print(f"  [skip] no rocSHMEM {op} data")
            continue
        plot_line(ax_lat, data["size"], data["latency_us"], color=color, marker=marker, label=label)
        plot_line(ax_bw, data["size"], data["bandwidth_gbs"], color=color, marker=marker, label=label)

    style_axis(ax_lat, "Message Size (Bytes)", "Latency (us)", "Latency")
    style_axis(ax_bw, "Message Size (Bytes)", "Bandwidth (GB/s)", "Bandwidth")

    finalize(fig, ax_lat, ax_bw, args.output, show=args.show)


if __name__ == "__main__":
    main()
