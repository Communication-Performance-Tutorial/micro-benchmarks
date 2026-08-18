# Plotting Scripts

Simple, self-contained Python scripts that turn the `.dat` files produced by `osu/run.sh`, `rccl-tests/run.sh`, and `rocshmem/run.sh` into latency/bandwidth figures. Every script produces one figure with two side-by-side subplots — **latency on the left, bandwidth (or busbw) on the right** — using a shared visual style so every plot looks consistent.

---

## File layout

```
plotting-scripts/
├── README.md               ← this file
├── requirements.txt         ← matplotlib, numpy
├── common_style.py          ← shared styling + .dat file parsers (not run directly)
├── osu_ccd_plot.py          ← OSU pt2pt: INTRA_CCD / INTER_CCD / INTER_SOCKET sweep (GPU buffers, D D)
├── osu_xcd_plot.py          ← OSU pt2pt: INTER_XCD (CPX mode) sweep (GPU buffers, D D)
├── osu_host_plot.py         ← OSU pt2pt: CPU affinity sweep (CPU/host buffers, H H)
├── rccl_sendrecv_plot.py    ← RCCL sendrecv_perf: INTER_SOCKET vs. INTER_XCD (CPX)
├── rccl_allreduce_plot.py   ← RCCL all_reduce_perf: latency + bus bandwidth
├── rocshmem_plot.py         ← rocSHMEM: Put vs. Get, latency + bandwidth
├── combined_inter_socket_plot.py  ← OSU + RCCL + rocSHMEM together, INTER_SOCKET only
└── plots/                   ← output PNGs land here by default
```

## Requirements

```bash
pip install -r requirements.txt
```

## Quick start

Run any script from this directory with no arguments — it reads from the neighboring suite's default `results/` folder and writes a PNG to `plots/`:

```bash
python osu_ccd_plot.py
python osu_xcd_plot.py
python osu_host_plot.py
python rccl_sendrecv_plot.py
python rccl_allreduce_plot.py
python rocshmem_plot.py
python combined_inter_socket_plot.py
```

Add `--show` to also pop up an interactive matplotlib window (useful locally; skip it on a headless login/compute node). Every script also accepts `--results-dir` and `--output` to point at a different results folder or output path.

Missing `.dat` files (e.g. you've only run SPX mode, not CPX) are skipped with a `[skip] ...` message instead of crashing the script — you always get a plot of whatever data is available.

---

## The scripts

### `osu_ccd_plot.py`

Reads `osu_latency_<mode>_<engine>.dat` / `osu_bw_<mode>_<engine>.dat` from `osu/results/` (SPX mode). Defaults to `INTRA_CCD`, `INTER_CCD`, `INTER_SOCKET` at the `SDMA` copy engine — one color per mode.

```bash
python osu_ccd_plot.py --modes INTRA_CCD INTER_CCD --engines SDMA BLIT
```

### `osu_xcd_plot.py`

Same file format, but targets the CPX-mode `INTER_XCD` sweep. Since CPX mode applies no CPU pinning, the interesting axis is the **copy engine** — defaults to comparing `SDMA` vs. `BLIT` for `INTER_XCD`.

```bash
python osu_xcd_plot.py --modes INTER_XCD INTER_SOCKET --engines SDMA
```

### `osu_host_plot.py`

Reads `osu_latency_H_<mode>.dat` / `osu_bw_H_<mode>.dat` from `osu/results/` (produced by `osu/run_host.sh`) — the pure CPU-buffer (`H H`) baseline, no GPU involved. No copy-engine axis (SDMA/BLIT only matters for GPU-side copies), so it's one line per affinity mode.

```bash
python osu_host_plot.py --modes INTRA_CCD INTER_CCD
```

### `rccl_sendrecv_plot.py`

Reads `sendrecv_INTER_SOCKET.dat` (SPX) and `sendrecv_cpx_INTER_XCD.dat` (CPX) from `rccl-tests/results/`. Plots out-of-place time and out-of-place algorithmic bandwidth.

### `rccl_allreduce_plot.py`

Globs `all_reduce*.dat` in `rccl-tests/results/` (so multiple runs with different GPU counts can be overlaid) and plots latency vs. **bus bandwidth** — the standard metric for comparing collective performance against the hardware peak. Each line is labeled with the GPU count parsed from its header.

```bash
python rccl_allreduce_plot.py --glob "all_reduce*.dat"
```

### `rocshmem_plot.py`

Reads `put_bw_<mode>.dat` / `get_bw_<mode>.dat` from `rocshmem/results/`. Defaults to all three affinity modes; color encodes mode, linestyle encodes operation (Put = solid, Get = dashed).

```bash
python rocshmem_plot.py --modes INTER_SOCKET
```

### `combined_inter_socket_plot.py`

Puts all three suites on one figure — one line per suite (OSU, RCCL, rocSHMEM) — using `INTER_SOCKET` (cross-package, over xGMI/Infinity Fabric), the only affinity mode all three suites can actually produce. `INTER_XCD` was considered but dropped: rocSHMEM's `run.sh` has no CPX/XCD-partitioning support, so it can't produce a genuine `INTER_XCD` data point (see the caveat in `rocshmem/README.md` — all of its CPU affinity modes are functionally `INTER_SOCKET`-class). Kept to three lines total for readability; rocSHMEM defaults to its Put data (pass `--rocshmem-op both` to add Get, or `--rocshmem-op get` to swap it).

```bash
python combined_inter_socket_plot.py --rocshmem-op both --osu-engine BLIT
```

---

## Shared style (`common_style.py`)

Every script above imports this module rather than duplicating formatting code:

- Grey dashed grid lines (`GRID_KW`) on both subplots.
- Log-log axes (message size spans several orders of magnitude in every suite).
- A fixed figure size and subplot spacing (`FIGSIZE`, `SUBPLOT_WSPACE`) shared by every script, so the two panels stay visually balanced and consistently spaced across all plots.
- A stable color-per-series legend via `SeriesStyler` — the first time a label (e.g. `INTER_SOCKET`) is plotted it's assigned the next color in the palette, and every subsequent line with that label reuses it.
- A linestyle cycle (`LINESTYLES`) for a secondary comparison axis (copy engine, Put vs. Get).
- `set_plain_y_ticks` — forces plain-number major y ticks (e.g. `50, 60, 70`) instead of log-scale power-of-ten notation, for axes whose data spans less than one decade.
- `load_osu`, `load_rccl`, `load_rocshmem` — parsers for each suite's `.dat` format that skip comment/header lines and return plain numpy arrays.

This module isn't meant to be run directly; import from it if you want to write a new plot with the same look.
