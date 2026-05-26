#!/usr/bin/env python3
"""Integrate the linearized barotropic QG equation in Mercator coordinates.

The implementation follows Mercator-QG-notes.tex:

    q_t + U/m q_x + V/m q_y + m^-2 J(psi, Q0)
        = -r q + kappa m^-2 Delta_M q + F,
    Delta_M psi = m^2 q,

where m = a cos(phi), x is longitude, y is the Mercator latitude, and the
diagnostic elliptic problem is solved with periodic longitude and homogeneous
Dirichlet streamfunction at the northern/southern boundaries.
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "Output"
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter, PillowWriter
from scipy.fft import dst, idst, irfft, rfft
from scipy.interpolate import RegularGridInterpolator


EARTH_RADIUS_M = 6_371_000.0
EARTH_ROTATION_S = 7.2921159e-5
SECONDS_PER_DAY = 86_400.0
SECONDS_PER_YEAR = 365.25 * SECONDS_PER_DAY


@dataclass(frozen=True)
class Grid:
    x: np.ndarray
    y: np.ndarray
    phi: np.ndarray
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    dx: float
    dy: float
    m: np.ndarray
    inv_m: np.ndarray
    inv_m2: np.ndarray


@dataclass(frozen=True)
class Background:
    u: np.ndarray
    v: np.ndarray
    q0: np.ndarray
    q0x: np.ndarray
    q0y: np.ndarray
    residual: np.ndarray


@dataclass(frozen=True)
class Inversion:
    denominator: np.ndarray


@dataclass(frozen=True)
class Config:
    input_path: Path
    output_path: Path
    years: float
    lon_resolution_deg: float
    lat_resolution_deg: float
    lat_bound_deg: float
    dt_seconds: float | None
    cfl: float
    snapshot_days: float
    fps: int
    friction_days: float
    kappa_m2_s: float
    forcing_amplitude: float
    forcing_mode: str
    forcing_period_days: float
    pulse_center_days: float
    pulse_sigma_days: float
    forcing_structure: str
    include_background_residual: bool
    zonal_mean_damping_days: float
    remove_zonal_mean: bool
    time_scheme: str
    plot_vmax: float | None
    dry_run: bool
    max_steps: int | None
    max_snapshots: int | None


def mercator_y(phi: np.ndarray) -> np.ndarray:
    return np.log(np.tan(np.pi / 4.0 + phi / 2.0))


def phi_from_mercator_y(y: np.ndarray) -> np.ndarray:
    return np.arcsin(np.tanh(y))


def build_grid(config: Config) -> Grid:
    nx = int(round(360.0 / config.lon_resolution_deg))
    ny = int(round(2.0 * config.lat_bound_deg / config.lat_resolution_deg)) + 1
    if nx < 4 or ny < 5:
        raise ValueError("Grid is too small for centered differences and inversion")

    lat_bound = np.deg2rad(config.lat_bound_deg)
    y_south = float(mercator_y(np.array([-lat_bound]))[0])
    y_north = float(mercator_y(np.array([lat_bound]))[0])

    x = np.linspace(0.0, 2.0 * np.pi, nx, endpoint=False)
    y = np.linspace(y_south, y_north, ny)
    phi = phi_from_mercator_y(y)

    dx = 2.0 * np.pi / nx
    dy = (y_north - y_south) / (ny - 1)
    cos_phi = np.cos(phi)
    m = EARTH_RADIUS_M * cos_phi

    return Grid(
        x=x,
        y=y,
        phi=phi,
        lon_deg=np.rad2deg(x),
        lat_deg=np.rad2deg(phi),
        dx=dx,
        dy=dy,
        m=m,
        inv_m=1.0 / m,
        inv_m2=1.0 / (m * m),
    )


def read_uv_file(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path, "r") as handle:
        lon = np.asarray(handle["lon"]).squeeze()
        lat = np.asarray(handle["lat"]).squeeze()
        u = np.asarray(handle["U_DJF_300hPa"])
        v = np.asarray(handle["V_DJF_300hPa"])

    if u.shape != (lat.size, lon.size) or v.shape != u.shape:
        raise ValueError(
            f"Unexpected input shapes: lon={lon.shape}, lat={lat.shape}, "
            f"U={u.shape}, V={v.shape}"
        )
    return lon, lat, u, v


def interpolate_periodic_lon(
    lon_src: np.ndarray,
    lat_src: np.ndarray,
    field: np.ndarray,
    lon_dst: np.ndarray,
    lat_dst: np.ndarray,
) -> np.ndarray:
    lon_ext = np.concatenate([lon_src, [lon_src[0] + 360.0]])
    field_ext = np.concatenate([field, field[:, :1]], axis=1)
    interpolator = RegularGridInterpolator(
        (lat_src, lon_ext),
        field_ext,
        bounds_error=False,
        fill_value=None,
    )
    lon_grid, lat_grid = np.meshgrid(lon_dst, lat_dst)
    points = np.column_stack([lat_grid.ravel(), lon_grid.ravel()])
    return interpolator(points).reshape(lat_dst.size, lon_dst.size)


def ddx_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1, axis=1) - np.roll(field, 1, axis=1)) / (2.0 * dx)


def ddy_centered(field: np.ndarray, dy: float) -> np.ndarray:
    out = np.empty_like(field)
    out[1:-1, :] = (field[2:, :] - field[:-2, :]) / (2.0 * dy)
    out[0, :] = (field[1, :] - field[0, :]) / dy
    out[-1, :] = (field[-1, :] - field[-2, :]) / dy
    return out


def d2dx_periodic(field: np.ndarray, dx: float) -> np.ndarray:
    return (np.roll(field, -1, axis=1) - 2.0 * field + np.roll(field, 1, axis=1)) / dx**2


def d2dy_centered(field: np.ndarray, dy: float) -> np.ndarray:
    out = np.empty_like(field)
    out[1:-1, :] = (field[2:, :] - 2.0 * field[1:-1, :] + field[:-2, :]) / dy**2
    out[0, :] = out[1, :]
    out[-1, :] = out[-2, :]
    return out


def load_background(config: Config, grid: Grid) -> Background:
    lon_src, lat_src, u_src, v_src = read_uv_file(config.input_path)
    u = interpolate_periodic_lon(lon_src, lat_src, u_src, grid.lon_deg, grid.lat_deg)
    v = interpolate_periodic_lon(lon_src, lat_src, v_src, grid.lon_deg, grid.lat_deg)

    ux = ddx_periodic(u, grid.dx)
    uy = ddy_centered(u, grid.dy)
    vx = ddx_periodic(v, grid.dx)

    sin_phi = np.sin(grid.phi)[:, None]
    cos_phi = np.cos(grid.phi)[:, None]
    f = 2.0 * EARTH_ROTATION_S * sin_phi
    zeta0 = (vx - uy + u * sin_phi) / (EARTH_RADIUS_M * cos_phi)
    q0 = f + zeta0

    q0x = ddx_periodic(q0, grid.dx)
    q0y = ddy_centered(q0, grid.dy)
    residual = (u * grid.inv_m[:, None]) * q0x + (v * grid.inv_m[:, None]) * q0y

    return Background(u=u, v=v, q0=q0, q0x=q0x, q0y=q0y, residual=residual)


def build_inversion(grid: Grid) -> Inversion:
    nx = grid.x.size
    ny_int = grid.y.size - 2
    kx = np.arange(nx // 2 + 1)
    ky = np.arange(1, ny_int + 1)
    lambda_x = -4.0 * np.sin(np.pi * kx / nx) ** 2 / grid.dx**2
    lambda_y = -4.0 * np.sin(np.pi * ky / (2.0 * (ny_int + 1))) ** 2 / grid.dy**2
    return Inversion(denominator=lambda_y[:, None] + lambda_x[None, :])


def invert_streamfunction(q: np.ndarray, grid: Grid, inversion: Inversion) -> np.ndarray:
    rhs = (grid.m[:, None] ** 2) * q
    rhs_int = rhs[1:-1, :]
    coeff = rfft(dst(rhs_int, type=1, axis=0, norm="ortho"), axis=1, norm="ortho")
    psi_coeff = coeff / inversion.denominator
    psi_int = idst(
        irfft(psi_coeff, n=grid.x.size, axis=1, norm="ortho"),
        type=1,
        axis=0,
        norm="ortho",
    )
    psi = np.zeros_like(q)
    psi[1:-1, :] = psi_int
    return psi


def normalize_pattern(pattern: np.ndarray) -> np.ndarray:
    scale = float(np.nanmax(np.abs(pattern)))
    if scale == 0.0:
        return pattern
    return pattern / scale


def forcing_pattern(config: Config, grid: Grid) -> np.ndarray:
    x_center = np.deg2rad(215.0)
    sigma_lambda = np.deg2rad(25.0)
    sigma_phi = np.deg2rad(5.0)
    dx_wrap = np.arctan2(np.sin(grid.x - x_center), np.cos(grid.x - x_center))
    gaussian = np.exp(
        -0.5
        * (
            (dx_wrap[None, :] / sigma_lambda) ** 2
            + (grid.phi[:, None] / sigma_phi) ** 2
        )
    )
    if config.forcing_structure == "gaussian":
        return gaussian

    weights = grid.m[:, None] ** 2
    if config.forcing_structure == "zero-mean-gaussian":
        weighted_mean = np.sum(gaussian * weights) / np.sum(weights)
        return normalize_pattern(gaussian - weighted_mean)

    if config.forcing_structure == "laplacian-gaussian":
        laplacian = d2dx_periodic(gaussian, grid.dx) + d2dy_centered(gaussian, grid.dy)
        weighted_mean = np.sum(laplacian * weights) / np.sum(weights)
        return normalize_pattern(laplacian - weighted_mean)

    if config.forcing_structure == "meridional-dipole":
        dipole = -(grid.phi[:, None] / sigma_phi) * gaussian
        weighted_mean = np.sum(dipole * weights) / np.sum(weights)
        return normalize_pattern(dipole - weighted_mean)

    raise ValueError(f"Unknown forcing structure {config.forcing_structure!r}")


def temporal_envelope(config: Config, t_seconds: float) -> float:
    if config.forcing_mode == "steady":
        return 1.0
    if config.forcing_mode == "cos":
        omega = 2.0 * np.pi / (config.forcing_period_days * SECONDS_PER_DAY)
        return float(np.cos(omega * t_seconds))
    if config.forcing_mode == "pulse":
        center = config.pulse_center_days * SECONDS_PER_DAY
        sigma = config.pulse_sigma_days * SECONDS_PER_DAY
        return float(np.exp(-0.5 * ((t_seconds - center) / sigma) ** 2))
    raise ValueError(f"Unknown forcing mode {config.forcing_mode!r}")


def make_rhs(
    config: Config,
    grid: Grid,
    background: Background,
    pattern: np.ndarray,
) -> Callable[[np.ndarray, np.ndarray, float], np.ndarray]:
    base_friction = 0.0 if config.friction_days <= 0.0 else 1.0 / (
        config.friction_days * SECONDS_PER_DAY
    )
    zonal_mean_damping = (
        0.0
        if config.zonal_mean_damping_days <= 0.0
        else 1.0 / (config.zonal_mean_damping_days * SECONDS_PER_DAY)
    )

    u_adv = background.u * grid.inv_m[:, None]
    v_adv = background.v * grid.inv_m[:, None]
    residual = background.residual if config.include_background_residual else 0.0

    def rhs(q: np.ndarray, psi: np.ndarray, t_seconds: float) -> np.ndarray:
        qx = ddx_periodic(q, grid.dx)
        qy = ddy_centered(q, grid.dy)
        psix = ddx_periodic(psi, grid.dx)
        psiy = ddy_centered(psi, grid.dy)
        jacobian = psix * background.q0y - psiy * background.q0x
        forcing = (
            config.forcing_amplitude
            * temporal_envelope(config, t_seconds)
            * pattern
        )
        biharmonic = config.kappa_m2_s * grid.inv_m2[:, None] * (
            d2dx_periodic(q, grid.dx) + d2dy_centered(q, grid.dy)
        )
        tendency = (
            -u_adv * qx
            - v_adv * qy
            - grid.inv_m2[:, None] * jacobian
            - base_friction * q
            + biharmonic
            - zonal_mean_damping * np.mean(q, axis=1, keepdims=True)
            + forcing
            - residual
        )
        tendency[0, :] = 0.0
        tendency[-1, :] = 0.0
        return tendency

    return rhs


def euler_step(
    q: np.ndarray,
    psi: np.ndarray,
    t_seconds: float,
    dt: float,
    rhs: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    grid: Grid,
    inversion: Inversion,
) -> tuple[np.ndarray, np.ndarray]:
    q_new = q + dt * rhs(q, psi, t_seconds)
    q_new[0, :] = 0.0
    q_new[-1, :] = 0.0
    return q_new, invert_streamfunction(q_new, grid, inversion)


def rk4_step(
    q: np.ndarray,
    _psi: np.ndarray,
    t_seconds: float,
    dt: float,
    rhs: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    grid: Grid,
    inversion: Inversion,
) -> tuple[np.ndarray, np.ndarray]:
    psi1 = invert_streamfunction(q, grid, inversion)
    k1 = rhs(q, psi1, t_seconds)

    q2 = q + 0.5 * dt * k1
    psi2 = invert_streamfunction(q2, grid, inversion)
    k2 = rhs(q2, psi2, t_seconds + 0.5 * dt)

    q3 = q + 0.5 * dt * k2
    psi3 = invert_streamfunction(q3, grid, inversion)
    k3 = rhs(q3, psi3, t_seconds + 0.5 * dt)

    q4 = q + dt * k3
    psi4 = invert_streamfunction(q4, grid, inversion)
    k4 = rhs(q4, psi4, t_seconds + dt)

    q_new = q + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
    q_new[0, :] = 0.0
    q_new[-1, :] = 0.0
    return q_new, invert_streamfunction(q_new, grid, inversion)


def estimate_dt(config: Config, grid: Grid, background: Background) -> float:
    if config.dt_seconds is not None:
        return config.dt_seconds

    u_coord = np.nanmax(np.abs(background.u * grid.inv_m[:, None]))
    v_coord = np.nanmax(np.abs(background.v * grid.inv_m[:, None]))
    advective_denominator = u_coord / grid.dx + v_coord / grid.dy
    candidates: list[float] = []
    if advective_denominator > 0.0:
        candidates.append(float(config.cfl / advective_denominator))

    if config.kappa_m2_s > 0.0:
        diffusion_denominator = (
            config.kappa_m2_s
            * np.nanmax(grid.inv_m2)
            * (1.0 / grid.dx**2 + 1.0 / grid.dy**2)
        )
        if diffusion_denominator > 0.0:
            candidates.append(float(0.45 / diffusion_denominator))

    if not candidates:
        return SECONDS_PER_DAY
    return min(candidates)


def cfl_number(dt: float, grid: Grid, background: Background) -> float:
    u_coord = np.nanmax(np.abs(background.u * grid.inv_m[:, None]))
    v_coord = np.nanmax(np.abs(background.v * grid.inv_m[:, None]))
    return float(dt * (u_coord / grid.dx + v_coord / grid.dy))


def format_duration(seconds: float) -> str:
    days = seconds / SECONDS_PER_DAY
    if days >= 1.0:
        return f"{days:.2f} days"
    hours = seconds / 3600.0
    if hours >= 1.0:
        return f"{hours:.2f} hours"
    return f"{seconds:.1f} seconds"


def make_writer(output_path: Path, fps: int):
    suffix = output_path.suffix.lower()
    if suffix in {".mp4", ".m4v", ".mov"}:
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            for candidate in (
                Path("/opt/homebrew/bin/ffmpeg"),
                Path("/usr/local/bin/ffmpeg"),
                Path("/opt/anaconda3/bin/ffmpeg"),
            ):
                if candidate.exists():
                    ffmpeg_path = str(candidate)
                    break
        if ffmpeg_path is None:
            raise RuntimeError(
                "MP4 output requires ffmpeg on PATH. Use a .gif output path "
                "with the available Pillow writer, or install ffmpeg."
            )
        matplotlib.rcParams["animation.ffmpeg_path"] = ffmpeg_path
        return FFMpegWriter(fps=fps, bitrate=2400)
    if suffix == ".gif":
        return PillowWriter(fps=fps)
    raise ValueError("Animation output must end in .mp4, .m4v, .mov, or .gif")


def plot_streamfunction(psi: np.ndarray, config: Config) -> np.ndarray:
    if config.remove_zonal_mean:
        return psi - np.mean(psi, axis=1, keepdims=True)
    return psi


def compute_total_kinetic_energy(psi: np.ndarray, grid: Grid) -> float:
    psix = ddx_periodic(psi, grid.dx)
    psiy = ddy_centered(psi, grid.dy)
    area = (grid.m[:, None] ** 2) * grid.dx * grid.dy
    return float(np.sum((psix**2 + psiy**2) * area))


def draw_animation(
    config: Config,
    grid: Grid,
    background: Background,
    inversion: Inversion,
    rhs: Callable[[np.ndarray, np.ndarray, float], np.ndarray],
    dt: float,
    ke_output: Path,
) -> None:
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except Exception as exc:
        raise RuntimeError("Cartopy is required for coastlines in the animation") from exc

    stepper = rk4_step if config.time_scheme == "rk4" else euler_step
    total_seconds = config.years * SECONDS_PER_YEAR
    total_steps = int(math.ceil(total_seconds / dt))
    if config.max_steps is not None:
        total_steps = min(total_steps, config.max_steps)

    snapshot_seconds = config.snapshot_days * SECONDS_PER_DAY
    next_snapshot = 0.0
    snapshot_count = 0
    ke_times: list[float] = []
    ke_values: list[float] = []

    lon_edges = np.rad2deg(np.linspace(0.0, 2.0 * np.pi, grid.x.size + 1))
    y_edges = np.linspace(
        grid.y[0] - 0.5 * grid.dy,
        grid.y[-1] + 0.5 * grid.dy,
        grid.y.size + 1,
    )
    lat_edges = np.rad2deg(phi_from_mercator_y(y_edges))

    q = np.zeros((grid.y.size, grid.x.size), dtype=np.float64)
    psi = np.zeros_like(q)

    fig = plt.figure(figsize=(13.5, 7.0), dpi=150)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.set_extent([0, 360, -config.lat_bound_deg, config.lat_bound_deg], ccrs.PlateCarree())
    ax.coastlines(linewidth=0.65, color="0.20")
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

    initial_vmax = config.plot_vmax if config.plot_vmax is not None else 1.0
    mesh = ax.pcolormesh(
        lon_edges,
        lat_edges,
        psi,
        transform=ccrs.PlateCarree(),
        cmap="RdBu_r",
        shading="flat",
        vmin=-initial_vmax,
        vmax=initial_vmax,
    )
    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.07, fraction=0.05)
    cbar.set_label("Perturbation streamfunction")
    title = ax.set_title("", fontsize=12, pad=11)
    writer = make_writer(config.output_path, config.fps)

    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    with writer.saving(fig, str(config.output_path), dpi=150):
        for step in range(total_steps + 1):
            t_seconds = step * dt
            if t_seconds + 0.5 * dt >= next_snapshot:
                psi_plot = plot_streamfunction(psi, config)
                ke_times.append(t_seconds / SECONDS_PER_DAY)
                ke_values.append(compute_total_kinetic_energy(psi, grid))
                if config.plot_vmax is None:
                    vmax = float(np.nanpercentile(np.abs(psi_plot), 99.0))
                    vmax = max(vmax, 1.0e-12)
                    mesh.set_clim(-vmax, vmax)
                    cbar.update_normal(mesh)
                mesh.set_array(psi_plot.ravel())
                title.set_text(
                    "Perturbation streamfunction"
                    + (" minus zonal mean, " if config.remove_zonal_mean else ", ")
                    + f"day {t_seconds / SECONDS_PER_DAY:.1f}"
                )
                writer.grab_frame()
                snapshot_count += 1
                next_snapshot += snapshot_seconds
                if (
                    config.max_snapshots is not None
                    and snapshot_count >= config.max_snapshots
                ):
                    break

            if step == total_steps:
                break

            q, psi = stepper(q, psi, t_seconds, dt, rhs, grid, inversion)
            if not np.all(np.isfinite(q)) or not np.all(np.isfinite(psi)):
                raise FloatingPointError(f"Non-finite model state at step {step + 1}")

            if step > 0 and step % max(1, total_steps // 20) == 0:
                print(
                    f"step {step}/{total_steps}, "
                    f"t={format_duration(t_seconds)}, "
                    f"|psi|max={np.nanmax(np.abs(psi)):.3e}",
                    flush=True,
                )

    plt.close(fig)
    if ke_times:
        fig_ke, ax_ke = plt.subplots(figsize=(9.5, 4.8), dpi=150)
        ax_ke.plot(ke_times, ke_values, color="#1f4e79", linewidth=1.5)
        ax_ke.set_xlabel("Days")
        ax_ke.set_ylabel("Total kinetic energy")
        ax_ke.set_title("Domain-integrated kinetic energy")
        ax_ke.grid(True, linewidth=0.3, alpha=0.5)
        fig_ke.tight_layout()
        ke_output.parent.mkdir(parents=True, exist_ok=True)
        fig_ke.savefig(ke_output, bbox_inches="tight")
        plt.close(fig_ke)
        print(f"Wrote {ke_output}")
    print(f"Wrote {config.output_path}")


def parse_dt(value: str) -> float | None:
    if value.lower() == "auto":
        return None
    return float(value)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(
        description="Integrate the Mercator-coordinate linearized barotropic QG equation."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "CESM2_piControl_DJF_UV_300hPa_climatology.mat",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR / "qg_streamfunction_1yr.mp4",
    )
    parser.add_argument("--years", type=float, default=1.0)
    parser.add_argument("--lon-resolution-deg", type=float, default=0.5)
    parser.add_argument("--lat-resolution-deg", type=float, default=0.5)
    parser.add_argument("--lat-bound-deg", type=float, default=80.0)
    parser.add_argument("--dt-seconds", type=parse_dt, default=None)
    parser.add_argument("--cfl", type=float, default=0.35)
    parser.add_argument("--snapshot-days", type=float, default=1.0)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--friction-days", type=float, default=20.0)
    parser.add_argument(
        "--kappa-m2-s",
        type=float,
        default=1.0e4,
        help="Coefficient for the +kappa nabla^4 psi term, in m^2/s.",
    )
    parser.add_argument("--forcing-amplitude", type=float, default=1.0e-11)
    parser.add_argument(
        "--forcing-mode",
        choices=("steady", "cos", "pulse"),
        default="steady",
    )
    parser.add_argument("--forcing-period-days", type=float, default=365.25)
    parser.add_argument("--pulse-center-days", type=float, default=30.0)
    parser.add_argument("--pulse-sigma-days", type=float, default=10.0)
    parser.add_argument(
        "--forcing-structure",
        choices=(
            "gaussian",
            "zero-mean-gaussian",
            "laplacian-gaussian",
            "meridional-dipole",
        ),
        default="gaussian",
    )
    parser.add_argument("--include-background-residual", action="store_true")
    parser.add_argument("--zonal-mean-damping-days", type=float, default=1.0 / 24.0)
    parser.add_argument("--remove-zonal-mean", action="store_true")
    parser.add_argument("--time-scheme", choices=("rk4", "euler"), default="rk4")
    parser.add_argument("--plot-vmax", type=float, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-snapshots", type=int, default=None)
    args = parser.parse_args()

    return Config(
        input_path=args.input,
        output_path=args.output,
        years=args.years,
        lon_resolution_deg=args.lon_resolution_deg,
        lat_resolution_deg=args.lat_resolution_deg,
        lat_bound_deg=args.lat_bound_deg,
        dt_seconds=args.dt_seconds,
        cfl=args.cfl,
        snapshot_days=args.snapshot_days,
        fps=args.fps,
        friction_days=args.friction_days,
        kappa_m2_s=args.kappa_m2_s,
        forcing_amplitude=args.forcing_amplitude,
        forcing_mode=args.forcing_mode,
        forcing_period_days=args.forcing_period_days,
        pulse_center_days=args.pulse_center_days,
        pulse_sigma_days=args.pulse_sigma_days,
        forcing_structure=args.forcing_structure,
        include_background_residual=args.include_background_residual,
        zonal_mean_damping_days=args.zonal_mean_damping_days,
        remove_zonal_mean=args.remove_zonal_mean,
        time_scheme=args.time_scheme,
        plot_vmax=args.plot_vmax,
        dry_run=args.dry_run,
        max_steps=args.max_steps,
        max_snapshots=args.max_snapshots,
    )


def main() -> int:
    config = parse_args()
    grid = build_grid(config)
    background = load_background(config, grid)
    inversion = build_inversion(grid)
    pattern = forcing_pattern(config, grid)
    rhs = make_rhs(config, grid, background, pattern)
    dt = estimate_dt(config, grid, background)
    total_seconds = config.years * SECONDS_PER_YEAR
    total_steps = int(math.ceil(total_seconds / dt))
    snapshots = int(math.floor(total_seconds / (config.snapshot_days * SECONDS_PER_DAY))) + 1

    print(f"Grid: nx={grid.x.size}, ny={grid.y.size}")
    print(
        "Latitude range: "
        f"{grid.lat_deg[0]:.3f} to {grid.lat_deg[-1]:.3f} degrees "
        "(uniform in Mercator y)"
    )
    print(f"dt={dt:.3f} s, advective CFL estimate={cfl_number(dt, grid, background):.3f}")
    print(f"Run length: {config.years:g} years, {total_steps} steps")
    print(f"Snapshots: {snapshots} every {config.snapshot_days:g} day(s), fps={config.fps}")
    print(f"Forcing structure: {config.forcing_structure}")
    print(f"Biharmonic coefficient kappa: {config.kappa_m2_s:.3e} m^2/s")
    if config.zonal_mean_damping_days > 0.0:
        print(
            "Zonal-mean streamfunction damping timescale: "
            f"{format_duration(config.zonal_mean_damping_days * SECONDS_PER_DAY)}"
        )
    if config.remove_zonal_mean:
        print("Plotting: streamfunction minus zonal mean")
    print(f"Output: {config.output_path}")

    if config.time_scheme == "euler":
        print(
            "Warning: forward Euler with centered advection is the literal "
            "scheme in the notes, but RK4 is normally more robust for long runs.",
            file=sys.stderr,
        )

    if config.dry_run:
        return 0

    ke_output = config.output_path.with_name(
        f"{config.output_path.stem}_kinetic_energy.png"
    )
    draw_animation(config, grid, background, inversion, rhs, dt, ke_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
