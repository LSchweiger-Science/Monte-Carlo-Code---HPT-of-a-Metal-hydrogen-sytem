# Monte Carlo Simulation of Metal-Hydrogen Systems under Severe Plastic Deformation

This repository contains a 2D Monte Carlo simulation designed to model the microstructural evolution of binary metal systems — with or without hydrogen — subjected to high-pressure torsion (HPT), a type of severe plastic deformation (SPD).

The simulation helps investigate:
- The effect of hydrogen on microstructure evolution during HPT.
- Whether mechanical alloying occurs or phase separation persists under shear deformation.
- The interplay between diffusion and shear, mimicking the atomic-scale mechanisms under SPD conditions.

Key Features:
- Models two-metal or two-metal + hydrogen systems.
- Incorporates both diffusion steps and shear steps:
  - Diffusion mimics atomic mobility.
  - Shear imitates the imposed plastic deformation.
- Requires user-defined diffusion and shear probabilities, which can be derived from physical parameters such as diffusion coefficients and strain rate.

Usage:

Main Code: Provided as a Spyder-compatible .py file containing:
- The full Monte Carlo simulation.
- Visualization and plotting tools for analyzing results.

Tutorial: A Jupyter Notebook is included to:
- Explain individual functions.
- Provide simple, illustrative examples.
- Help new users understand and adapt the code for their needs.
