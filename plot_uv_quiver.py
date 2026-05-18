#!/usr/bin/env python3
"""Plot a quiver map from the CESM2 300 hPa DJF U/V climatology file."""

from __future__ import annotations

import argparse
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Output"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_uv(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with h5py.File(path, "r") as handle:
        lon = np.asarray(handle["lon"]).squeeze()
        lat = np.asarray(handle["lat"]).squeeze()
        u = np.asarray(handle["U_DJF_300hPa"])
        v = np.asarray(handle["V_DJF_300hPa"])
        plev = float(np.asarray(handle["plev_target"]).squeeze())

    if u.shape != (lat.size, lon.size) or v.shape != u.shape:
        raise ValueError(
            f"Unexpected shapes: lon={lon.shape}, lat={lat.shape}, "
            f"U={u.shape}, V={v.shape}"
        )

    return lon, lat, u, v, plev


def plot_quiver(input_path: Path, output_path: Path, stride: int) -> None:
    lon, lat, u, v, plev = load_uv(input_path)
    lon_grid, lat_grid = np.meshgrid(lon, lat)

    row_slice = slice(None, None, stride)
    col_slice = slice(None, None, stride)
    lon_s = lon_grid[row_slice, col_slice]
    lat_s = lat_grid[row_slice, col_slice]
    u_s = u[row_slice, col_slice]
    v_s = v[row_slice, col_slice]
    speed_s = np.hypot(u_s, v_s)

    fig = plt.figure(figsize=(13, 6.5), dpi=180)

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature

        ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
        ax.set_global()
        ax.coastlines(linewidth=0.6, color="0.25")
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, edgecolor="0.45")
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.25,
            color="0.65",
            alpha=0.6,
            linestyle="-",
        )
        gl.top_labels = False
        gl.right_labels = False

        quiver = ax.quiver(
            lon_s,
            lat_s,
            u_s,
            v_s,
            speed_s,
            transform=ccrs.PlateCarree(),
            cmap="viridis",
            scale=620,
            width=0.0022,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=3.8,
            pivot="middle",
        )
    except Exception as exc:
        print(f"Cartopy map unavailable, falling back to lon/lat axes: {exc}")
        ax = plt.axes()
        ax.set_xlim(float(lon.min()), float(lon.max()))
        ax.set_ylim(float(lat.min()), float(lat.max()))
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(linewidth=0.25, color="0.75")
        quiver = ax.quiver(
            lon_s,
            lat_s,
            u_s,
            v_s,
            speed_s,
            cmap="viridis",
            scale=620,
            width=0.0022,
            headwidth=3.5,
            headlength=4.5,
            headaxislength=3.8,
            pivot="middle",
        )

    ax.set_title(
        f"CESM2 piControl DJF wind climatology at {plev / 100:.0f} hPa",
        fontsize=13,
        pad=12,
    )
    ax.quiverkey(quiver, 0.88, -0.08, 20, "20 m/s", labelpos="E")

    cbar = fig.colorbar(quiver, ax=ax, orientation="horizontal", pad=0.07, fraction=0.045)
    cbar.set_label("Wind speed (m/s)")

    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "CESM2_piControl_DJF_UV_300hPa_climatology.mat",
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR / "uv_quiver_300hPa.png")
    parser.add_argument("--stride", type=int, default=8)
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be at least 1")

    plot_quiver(args.input, args.output, args.stride)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
