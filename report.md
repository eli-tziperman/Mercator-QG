# Mercator QG Integration Attempt

## Goal

The requested task was to follow `codex-instructions.txt`: use the QG notes and
the provided CESM2 DJF 300 hPa `U,V` climatology to integrate the linearized
barotropic QG equation in Mercator coordinates, starting from zero stream
function, and create an animation of streamfunction evolution with continental
outlines.

## Input Data

- `CESM2_piControl_DJF_UV_300hPa_climatology.mat`
- Variables found in the MATLAB/HDF5 file:
  - `lon`
  - `lat`
  - `U_DJF_300hPa`
  - `V_DJF_300hPa`
  - `plev_target`
  - `N`

Before the integration work, I also made a quiver plot of the input wind field:

- `plot_uv_quiver.py`
- `uv_quiver_300hPa.png`

## Main Integration Script

I created:

- `integrate_mercator_qg.py`

The script implements the prognostic-diagnostic system from
`Mercator-QG-notes.tex`:

```text
q_t + U/m q_x + V/m q_y + m^-2 J(psi, Q0) = -r q + F
Delta_M psi = m^2 q
```

where:

- `q` is perturbation relative vorticity/PV anomaly.
- `psi` is perturbation streamfunction.
- `m = a cos(phi)`.
- `Q0 = f + zeta0` is the background absolute vorticity.
- Longitude is periodic.
- Meridional streamfunction boundary condition is homogeneous Dirichlet at
  the artificial north/south boundaries.

Implementation details:

- Grid: 0.5 degree nominal resolution, `720 x 321`.
- Latitude domain: 80S to 80N.
- Grid is uniform in Mercator `y`, not uniform in physical latitude.
- Input CESM winds are interpolated to the model grid.
- Background PV and PV gradients are computed by finite differences.
- PV is advanced with RK4 by default.
- The streamfunction inversion uses FFT in longitude and sine transforms in
  the meridional direction.
- Cartopy is used for coastlines and borders.
- MP4 output uses `/opt/homebrew/bin/ffmpeg` if `ffmpeg` is not on PATH.

## Original One-Year Run

I changed the default run from 10 years to 1 year after `ffmpeg` was installed,
then ran the full simulation:

```bash
/opt/anaconda3/bin/python integrate_mercator_qg.py
```

Output:

- `qg_streamfunction_1yr.mp4`
- `qg_streamfunction_1yr_lastframe.png`

Video metadata from `ffprobe`:

- Resolution: `2024 x 1050`
- FPS: `30`
- Frames: `366`
- Duration: `12.2 s`

The run completed successfully. The final logged model time was about one year,
and the final maximum streamfunction magnitude was about `1.49e9`.

## Problem With The Original Output

The resulting animation did not look like the expected alternating
positive-negative wave train from the Nino3.4 region.

Observed issues:

- The streamfunction response was mostly one-signed.
- Positive anomalies were much weaker than negative anomalies.
- A large accumulated signal appeared north of 60N.

## Diagnosis

I concluded that the original setup was not a clean wave-train diagnostic.

Main reasons:

1. The forcing was a one-signed Gaussian PV tendency.

   A one-signed PV source in

   ```text
   Delta_M psi = m^2 q
   ```

   naturally inverts to a mostly one-signed streamfunction response. So the
   sign imbalance is not surprising.

2. The animation plotted raw streamfunction.

   Raw `psi` contains large-scale and zonal-mean components. These can dominate
   the color scale and hide the alternating wave component.

3. The model had hard meridional walls at +/-80 degrees.

   With steady forcing and no high-latitude sponge, the model can accumulate or
   trap high-latitude structure through boundary effects, free modes, or
   waveguide behavior.

## Diagnostic Updates Added

I added optional controls to `integrate_mercator_qg.py`:

- `--forcing-structure gaussian`
- `--forcing-structure zero-mean-gaussian`
- `--forcing-structure laplacian-gaussian`
- `--forcing-structure meridional-dipole`
- `--remove-zonal-mean`
- `--sponge-lat-deg`
- `--sponge-days`
- `--sponge-power`

The baseline defaults remain reproducible, but these options make it possible
to run more wave-focused diagnostics.

## Preview Runs

I ran a short diagnostic preview using a zero-net Laplacian-Gaussian forcing,
zonal-mean removal in the plotted streamfunction, and a high-latitude sponge:

```bash
/opt/anaconda3/bin/python integrate_mercator_qg.py \
  --years 0.1 \
  --forcing-structure laplacian-gaussian \
  --remove-zonal-mean \
  --sponge-lat-deg 60 \
  --output qg_wave_diagnostic_preview.mp4
```

Output:

- `qg_wave_diagnostic_preview.mp4`
- `qg_wave_diagnostic_preview_lastframe.png`

This preview showed more alternating positive-negative structure, but still had
notable high-latitude features.

I then ran a stronger sponge/friction preview:

```bash
/opt/anaconda3/bin/python integrate_mercator_qg.py \
  --years 0.1 \
  --forcing-structure laplacian-gaussian \
  --remove-zonal-mean \
  --sponge-lat-deg 45 \
  --sponge-days 1 \
  --friction-days 10 \
  --output qg_wave_diagnostic_preview_strong_sponge.mp4
```

Output:

- `qg_wave_diagnostic_preview_strong_sponge.mp4`
- `qg_wave_diagnostic_preview_strong_sponge_lastframe.png`

This stronger sponge case leveled off rather than continuing to accumulate, but
it still showed significant high-latitude structure.

## Conclusions

The original one-year animation should not be interpreted as a clean physical
Nino3.4 Rossby wave-train response. It is strongly affected by the one-signed
PV forcing, raw streamfunction plotting, and high-latitude boundary behavior.

For a better wave-train diagnostic, the next production run should use:

```bash
/opt/anaconda3/bin/python integrate_mercator_qg.py \
  --forcing-structure laplacian-gaussian \
  --remove-zonal-mean \
  --sponge-lat-deg 45 \
  --sponge-days 1 \
  --friction-days 10 \
  --output qg_wave_diagnostic_1yr.mp4
```

Even with these changes, the high-latitude response should be treated
cautiously. A more robust physical setup may need one or more of:

- A forcing that represents heating/divergence more directly rather than a
  one-signed barotropic PV source.
- Better meridional boundary treatment.
- A sponge layer tuned less aggressively but over a wider latitude band.
- A check of the background PV gradients and waveguide structure.
- Possibly plotting vorticity or geopotential-like anomalies instead of raw
  streamfunction.

## Should the +/-80 Degree Restriction Be Removed?

No, the latitude restriction should not simply be removed all the way to the
poles.

In this Mercator formulation, the poles are singular:

```text
y = log tan(pi/4 + phi/2)
m = a cos(phi)
```

As `phi` approaches +/-90 degrees:

- `y` approaches +/-infinity.
- `cos(phi)` approaches zero.
- `1/m` and `1/m^2` become very large.
- The advection and PV-gradient terms become poorly conditioned.
- The time step becomes more restrictive.
- Polar artifacts usually get worse, not better.

Therefore, removing the +/-80 degree restriction is not the right fix for the
high-latitude accumulation.

A better approach is:

1. Keep a finite latitude boundary.
2. Add a sponge layer before the boundary, for example starting at 45 or
   60 degrees.
3. Plot `psi` with the zonal mean removed when diagnosing wave trains.
4. Run sensitivity tests with boundaries such as +/-65, +/-70, +/-75, and
   +/-80 degrees, and check whether the solution below about 60 degrees changes.

For the Nino3.4 wave-train diagnostic, a reasonable test command is:

```bash
/opt/anaconda3/bin/python integrate_mercator_qg.py \
  --lat-bound-deg 75 \
  --forcing-structure laplacian-gaussian \
  --remove-zonal-mean \
  --sponge-lat-deg 50 \
  --sponge-days 2 \
  --friction-days 10 \
  --output qg_wave_diagnostic_75deg_1yr.mp4
```

If polar or fully global behavior is required, this Mercator finite-difference
setup is not the right grid. A better choice would be a spherical harmonic
model, a cubed-sphere grid, a latitude-longitude model with explicit polar
treatment, or another non-singular global discretization.
