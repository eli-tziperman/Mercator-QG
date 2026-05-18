# Instructions for creating the notes for this project

1. Write the linearized barotropic potential-vorticity equation in terms of the stream function in Mercator coordinates, with a steady background flow whose spherical-coordinate winds are
   \[
   (U(\lambda,\phi),V(\lambda,\ \phi)).
   \]

2. Add:
   - A friction term proportional to
     \[
     -r \nabla^2\ \psi,
     \]
   - A steady forcing term that is a Gaussian in latitude and longitude in the Niño 3.4 region.
   - The two-step time-advancement procedure:
     1. first advance the PV,
     2. then solve the diagnostic equation for the stream function.

3. Add the finite-difference formulation in both time and space as a new section.

4. Make the domain be from 80S to 80N, all longitudes, with periodic boundary conditions in longitude.

5. Give the result as a complete LaTeX file that can be compiled.

