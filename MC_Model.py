# -*- coding: utf-8 -*-
"""
Created on Wed Oct 16 14:30:32 2024

@author: lschweiger
"""

#%% IMPORT PACKAGES

import numpy as np

import random

import pandas as pd

import os  # For folder creation

import matplotlib.pyplot as plt

from matplotlib.colors import LinearSegmentedColormap, Normalize

from matplotlib.colors import ListedColormap

import matplotlib.colors as mcolors

from matplotlib import ticker

from scipy import constants

import seaborn as sns

from tqdm import tqdm  # For progress bar

from PIL import Image

from glob import glob

import cv2

import re

from numba import njit, prange

import time

# Defining the inch to cm ratio for plotting 
cm=1/2.54

#%% FUNCTIONS - MC SIMULATION


########################## METAL SYSTEM ######################################

# Function to initialize the grid with quadrants (checkerboard-like pattern with 4 fields)
def initialize_metal_grid(grid_size):
    """
    This defines a function named initialize_metal_grid().
    The purpose of this function is to initialize a metal grid with a checkerboard-like pattern
    consisting of four quadrants, representing two different types of atoms A (=0) and B (=1).
    A (=0) is the metal with high hydrogen affinity, B (=1) the atom with low hydrogen affinity.
    
    Parameters:
    - grid_size          : int, the size of the desired grid (e.g., 100 for a 100x100 grid).
    
    Returns:
    - metal_grid         : 2D numpy array, where 0 represents one type of atom (A) and 1 represents another (B).
    """
    
    # Creates a 2D array of size grid_size x grid_size, filled with zeros.
    metal_grid = np.zeros((grid_size, grid_size), np.int64)
    
    # This calculates half_size as half of the total grid size (grid_size), using integer division (//).
    half_size = grid_size // 2
    
    # Assigns 0 (representing A atoms) and 1 (representing B atoms) to the top-left quadrant of the grid.
    metal_grid[:half_size, :half_size] = 0  # Top-left quadrant (A atoms)
    metal_grid[:half_size, half_size:] = 1  # Top-right quadrant (B atoms)
    metal_grid[half_size:, :half_size] = 1  # Bottom-left quadrant (B atoms)
    metal_grid[half_size:, half_size:] = 0  # Bottom-right quadrant (A atoms)
    
    return metal_grid

# Alternatively we can generate a grid from any binary image derived from a real SEM image
def initialize_grid_from_image(image_path, grid_size):
    """
    Initializes a grid based on a binary input image. 
    Such a binary imput image can be a "thresholded" SEM image of a real microstructure.
    
    Parameters:
    - image_path           : str, path to the binary image file (black-and-white).
    - grid_size            : int, the size of the desired grid (e.g., 100 for a 100x100 grid).
    
    Returns:
    - metal_grid           : 2D numpy array, where 0 represents one type of atom and 1 represents another.
    """
    # Load the image in grayscale mode and convert it to a binary (black-and-white) image (this is done by .convert("L"))
    img = Image.open(image_path).convert("L")
    
    # Resize the image to the desired grid size, using antialiasing to improve downscaling
    img_resized = img.resize((grid_size, grid_size), Image.LANCZOS)
    
    # Convert image to binary values (0 or 1) based on a threshold (128 is the midpoint of grayscale)
    metal_grid = np.array(img_resized) > 128
    metal_grid = metal_grid.astype(np.int64)  # Convert to integer (0s and 1s)
    
    return metal_grid

# Function to perform a shear step (mechanical intermixing) with flow strength weighting
@njit(fastmath=True, cache=True)
def shear_step(metal_grid, shear_type, strength_A, strength_B, shear_counts_horizontal, shear_counts_vertical, max_attempts):
    """
    Performs a shear step on a grid with probability weighted by the cumulative strength of atoms along the shear line.
    The lower the total strength, the higher the probability for the shear step to occur.
    
    Parameters:
    - metal_grid                 : 2D numpy array, the grid of atoms.
    - shear_type                 : str, specifies the type of shear to apply:
                                    - 0 for horizontal shear only
                                    - 1 for vertical shear only
                                    - 2 for randomly selecting between horizontal and vertical (default).
    - strength_A                 : float, "flow strength" or "Hardness" of atom type A.
    - strength_B                 : float, "flow strength" of "Hardness" atom type B.
    - shear_counts_horizontal    : np.array, for saving the locations where shear steps were already performed
    - shear_counts_vertical      : np.array, for saving the locations where shear steps were already performed
    - max_attempts               : int, maximum number of iterations until which shear is enforced, avoid while loop for compiling.
    
    Returns:
    - None
    """

    grid_size_x, grid_size_y = metal_grid.shape

    # Determine shear direction (shear_type: 0 → horizontal, 1 → vertical, both → vertical)
    if shear_type == 2:
        horizontal_shear = np.random.rand() < 0.8  # 50% chance of horizontal shear
    else:
        horizontal_shear = shear_type == 0


    # Compute max strength once to avoid redundant calculations
    # Precompute strength for each row & column (vectorized)
    # strength_grid = (metal_grid == 0) * strength_A + (metal_grid == 1) * strength_B
    # row_strengths = np.sum(strength_grid, axis=1)  # Sum over columns (horizontal shear)
    # col_strengths = np.sum(strength_grid, axis=0)  # Sum over rows (vertical shear)

    # Maximum column / row strength normalization factor
    max_strength = 2 * max(strength_A, strength_B) * (grid_size_y if horizontal_shear else grid_size_x)
    
    # We must enforce a successful shear step - Therefore we track if shear has already been successful
    attempts = 0

    for _ in range(max_attempts):
        
        if horizontal_shear:
            # Select a random row
            line = np.random.randint(0, grid_size_x)
            
            # Compute cumulative strength (current + above row)
            cumulative_strength = np.sum((metal_grid[line, :] == 0) * strength_A + (metal_grid[line, :] == 1) * strength_B) + \
                                  np.sum((metal_grid[(line - 1) % grid_size_x, :] == 0) * strength_A + (metal_grid[(line - 1) % grid_size_x, :] == 1) * strength_B)
            
            # Compute shear probability
            shear_probability = 1.0 - (cumulative_strength / max_strength)
                        
            # Perform shear step if probability condition is met - Enforce a shear step after a max. number of attempts
            if np.random.rand() < shear_probability or _ == max_attempts - 1:
                
                last_col = metal_grid[:line, -1].copy()  # Store last column
                metal_grid[:line, 1:] = metal_grid[:line, :-1]  # Shift right
                metal_grid[:line, 0] = last_col  # Wrap-around
                
                shear_counts_horizontal[line] += 1  
                return 

        else:
            # Select a random column
            line = np.random.randint(0, grid_size_y)
            
            # Compute cumulative strength (current + left column)
            cumulative_strength = np.sum((metal_grid[:, line] == 0) * strength_A + (metal_grid[:, line] == 1) * strength_B) + \
                                  np.sum((metal_grid[:, (line - 1) % grid_size_y] == 0) * strength_A + (metal_grid[:, (line - 1) % grid_size_y] == 1) * strength_B)
                                  
            # Compute shear probability
            shear_probability = 1.0 - (cumulative_strength / max_strength)

            # Perform shear step if probability condition is met - Enforce a shear step after a max. number of attempts
            if np.random.rand() < shear_probability or attempts == max_attempts - 1:
                
                # Shift all columns left of the selected line downward
                last_row = metal_grid[-1, :line].copy()  # Store last row
                metal_grid[1:, :line] = metal_grid[:-1, :line]  # Shift down
                metal_grid[0, :line] = last_row  # Wrap-around
                
                shear_counts_vertical[line] += 1  
                return 
                
# Function to perform one Monte Carlo step (diffusion)
@njit(fastmath=True, cache=True)  # JIT compile for speed, use fastmath optimizations
def metal_diffusion_step(metal_grid, kT, E_AA, E_BB, E_AB, grid_size):
    """
    Perform a single MC step by trying to swap two neighboring atoms
    This defines a function called metal_diffusion_step(), which takes
    four arguments:

    Parameters:
    - metal_grid               : np.array, 2D array representing the atom positions (either A or B).
    - kT                 : float, thermal energy factor (k is the Boltzmann constant, T is temperature). It controls how likely energetically unfavorable moves are accepted (higher temperatures allow more such moves).
    - E_AA               : float, interaction energy between A-A atoms.
    - E_BB               : float, interaction energy between B-B atoms.
    - E_AB               : float, interaction energy between A-B atoms.
    
    Returns:
    - accept             : True if the MC step was accepted
    """

    # Choose a random position on the metal_grid
    x, y = np.random.randint(0, grid_size), np.random.randint(0, grid_size)

    # Select a random neighbor (up, down, left, right)
    direction = np.random.randint(0, 4)
    if direction == 0: nx, ny = (x - 1) % grid_size, y  # Up
    elif direction == 1: nx, ny = (x + 1) % grid_size, y  # Down
    elif direction == 2: nx, ny = x, (y - 1) % grid_size  # Left
    else: nx, ny = x, (y + 1) % grid_size  # Right
    
    if metal_grid[x, y] == metal_grid[nx, ny]:
        return True

    # Calculate the initial energy before swap
    initial_energy = calculate_metal_metal_interaction_energy(x,   y, metal_grid, E_AA, E_BB, E_AB) + \
                     calculate_metal_metal_interaction_energy(nx, ny, metal_grid, E_AA, E_BB, E_AB)
    
    # Swap atoms
    metal_grid[x, y], metal_grid[nx, ny] = metal_grid[nx, ny], metal_grid[x, y]
    
    # Calculate the new energy after swap
    final_energy = calculate_metal_metal_interaction_energy(x,   y, metal_grid, E_AA, E_BB, E_AB) + \
                   calculate_metal_metal_interaction_energy(nx, ny, metal_grid, E_AA, E_BB, E_AB)
    
    # Calculate the change in energy
    delta_E = final_energy - initial_energy
    
    accept = True
    # Metropolis criterion: accept swap with probability exp(-delta_E / kT)
    if delta_E > 0 and np.exp(-delta_E / kT) < np.random.rand():
        # Reject the swap (reverse it)
        metal_grid[x, y], metal_grid[nx, ny] = metal_grid[nx, ny], metal_grid[x, y]
        accept = False
        
    return accept

# Function to calculate local energy around a site (x, y), considering periodic boundary conditions
@njit(fastmath=True, cache=True)  # JIT compile for speed, use fastmath optimizations
def calculate_metal_metal_interaction_energy(x, y, metal_grid, E_AA, E_BB, E_AB):
    """
    The energy is calculated based on the type of the neighboring atom:
    If the neighboring atom is the same type as the current atom, the corresponding interaction energy (E_AA or E_BB) is added.
    If the neighboring atom is a different type, the E_AB interaction energy is added.
    
    Parameters
    - x               : int, x-position of site where to calculate the energy.
    - y               : int, y-position of site where to calculate the energy.
    - grid            : np.array, 2D array representing the atom positions (0 for A, 1 for B).
    - E_AA            : float, interaction energy between A-A atoms.
    - E_BB            : float, interaction energy between B-B atoms.
    - E_AB            : float, interaction energy between A-B atoms.
    
    Returns:
    - energy          : float, interaction energy derived from site (x, y).
    """
    energy = 0
    atom = metal_grid[x, y]  # Atom type at the target site (0 for A, 1 for B)
    
    # Define neighbors (up, down, left, right)
    neighbors = [(x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)]
    
    for nx, ny in neighbors:
        # Apply periodic boundary conditions to neighbors
        nx = nx % metal_grid.shape[0]  # Wrap around the x-coordinate
        ny = ny % metal_grid.shape[1]  # Wrap around the y-coordinate
        
        # Get the neighboring atom after applying periodic boundary conditions
        neighbor_atom = metal_grid[nx, ny]
        
        # Calculate the energy contribution based on the interaction type
        if atom == 0 and neighbor_atom == 0:  # A-A interaction
            energy += E_AA
        elif atom == 1 and neighbor_atom == 1:  # B-B interaction
            energy += E_BB
        else:  # A-B or B-A interaction
            energy += E_AB
    
    return energy

# Function to calculate total energy of the grid, considering periodic boundary conditions
def calculate_total_metal_metal_interaction_energy(metal_grid, E_AA, E_BB, E_AB):
    """
    Vectorized calculation of the total energy of the grid, 
    considering interactions with right and down neighbors only.
    
    Parameters:
    - metal_grid         : np.array, 2D array representing the atom positions (0 for A, 1 for B).
    - E_AA               : float, interaction energy between A-A atoms.
    - E_BB               : float, interaction energy between B-B atoms.
    - E_AB               : float, interaction energy between A-B atoms.
    
    Returns:
    - total_energy       : float, interaction energy summed up over the whole grid.
    """
    # Shifted grids for right and down neighbors, with periodic boundaries
    right_neighbors = np.roll(metal_grid, shift=-1, axis=1)  # Right neighbors
    down_neighbors = np.roll(metal_grid, shift=-1, axis=0)   # Down neighbors

    # Calculate interactions for right neighbors
    interaction_AA_right = (metal_grid == 0) & (right_neighbors == 0)  # A-A interactions
    interaction_BB_right = (metal_grid == 1) & (right_neighbors == 1)  # B-B interactions
    interaction_AB_right = (metal_grid != right_neighbors)             # A-B interactions

    # Calculate interactions for down neighbors
    interaction_AA_down = (metal_grid == 0) & (down_neighbors == 0)    # A-A interactions
    interaction_BB_down = (metal_grid == 1) & (down_neighbors == 1)    # B-B interactions
    interaction_AB_down = (metal_grid != down_neighbors)               # A-B interactions

    # Total energy contributions
    total_energy_right = (
        np.sum(interaction_AA_right) * E_AA +
        np.sum(interaction_BB_right) * E_BB +
        np.sum(interaction_AB_right) * E_AB
    )

    total_energy_down = (
        np.sum(interaction_AA_down) * E_AA +
        np.sum(interaction_BB_down) * E_BB +
        np.sum(interaction_AB_down) * E_AB
    )

    # Sum total energy contributions
    total_energy = total_energy_right + total_energy_down

    return total_energy


# Function to calculate the mixing index of the grid, considering periodic boundary conditions
def calculate_mixing_index(grid):
    """
    Vectorized calculation of the Mixing Index.
    The Mixing Index is the proportion of different neighboring pairs (A-B pairs).
    0   - Perfect seperation, no A-B neighbors
    0.5 - random solution
    1   - Perfect intermixing, only A-B neighbors
    
    Parameters:
    - grid                : np.array, 2D array representing the atom positions (either A or B).
       
    Returns:
    - mixing index        : float, represents the state of mixing / order of the current grid
    """
    # Shifted grids for right and down neighbors, with periodic boundaries
    right_neighbors = np.roll(grid, shift=-1, axis=1)
    down_neighbors = np.roll(grid, shift=-1, axis=0)

    # Calculate A-B pairs
    ab_pairs_right = grid != right_neighbors
    ab_pairs_down = grid != down_neighbors

    # Total pairs are grid_size^2 for both right and down neighbors (no double-counting)
    total_pairs = 2 * grid.size
    ab_pairs = np.sum(ab_pairs_right) + np.sum(ab_pairs_down)

    return ab_pairs / total_pairs

# Function to sample energy values from a normal distribution
def sample_energy(E_mean, E_std):
    """Function to sample interaction energies from a normal distribution
    
    Parameters:
        
    - E_mean          : float, mean value of the normal distribution
    - E_std           : float, Standard deviation of the normal distribution
    
    Returns:
    - E               : float, energy sample drawn form a normal distribution
    """
    return np.random.normal(E_mean, E_std)

# To make the frequency of energy and mixing index calculations dependent on the total number of Monte Carlo (MC) steps,
# you can adjust the interval dynamically based on the total `n_mc_steps`. Here's an implementation of that:
@njit(fastmath=True, cache=True)  # JIT compile for speed, use fastmath optimizations
def get_sampling_interval(n_mc_steps):
    """
    Function to determine the sampling interval for calculating energy and mixing index
    based on the total number of MC steps.
    
    The higher the number of MC steps, the less frequent the sampling will be.
    
    Parameters:
    - n_mc_steps           : int, current MC step
    
    Returns:
    - sampling_interval    : int, sampling interval for saving data and checking convergence based on current MC step
    """
    # Define a scaling factor or logic to decrease sampling frequency as n_mc_steps increases
    if n_mc_steps   <= 100:
        return 5      # Sample every 5 steps for smaller simulations
    if n_mc_steps   <= 1000:
        return 50     # Sample every 50 steps for smaller simulations
    elif n_mc_steps <= 10000:
        return 500    # Sample every 500 steps for mid-range simulations
    elif n_mc_steps <= 100000:
        return 5000   # Sample every 5000 steps for larger simulations
    elif n_mc_steps <= 1000000:
        return 10000  # Sample every 50000 steps for larger simulations
    else:
        return 25000  # Sample every 100000 steps for larger simulations

# Adjust the tick margins for the axis       
def adjust_margins(limits, ax, percent=0.02, scaling="linear"):

    # Get the position (bounding box) of the axis
    axis_position = ax.get_position()
    
    # Extract the width and height from the bounding box
    axis_width = axis_position.width
    axis_height = axis_position.height
    
    # Calculate the axis box ratio
    axis_box_ratio = axis_width / axis_height
    
    percent_y = axis_box_ratio*percent
    if scaling=="linear":
        limits = limits[0] - percent*abs(limits[1]-limits[0]), limits[1] + percent_y*abs(limits[1]-limits[0])
    elif scaling == "log":
        limits = [np.log10(limits[0]), np.log10(limits[1])]
        limits = 10**np.array(limits[0] - percent*abs(limits[1]-limits[0])), 10**np.array(limits[1] + percent_y*abs(limits[1]-limits[0]))
    return limits


##################### METAL-HYDROGEN SYSTEM ##################################

# Function to initialize the hydrogen grid (H atoms near A atoms)
def initialize_hydrogen_grid(grid_size):
    """
    The hydrogen grid will have the same size as the metal grid.
    We will place hydrogen atoms near A atoms, but ensure that the
    hydrogen atoms occupy interstitial sites at the boundary as well
    (effectively wrapping the grid).
    
    Parameters: 
    - grid_size:          int, size of the hydrogen grid, should be the same as that of the metal grid
    
    Returns
    - hydrogen_grid       2D np.array, grid representing the hydrogen atoms, 0 is a vacancy, 1 is a hydrogen atom
    """
    # Initialize a grid of the same size as the metal grid
    hydrogen_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
    
    # Place hydrogen near A atoms, with periodic boundary conditions
    # Top-left quadrant (A atoms)
    hydrogen_grid[:grid_size // 2, :grid_size // 2] = 1  # Hydrogen near A atoms (Top-left)
    
    # Bottom-right quadrant (A atoms)
    hydrogen_grid[grid_size // 2:, grid_size // 2:] = 1  # Hydrogen near A atoms (Bottom-right)
    
    return hydrogen_grid

# Function to perform a shear step (mechanical intermixing) with flow strength weighting
@njit(fastmath=True, cache=True)
def shear_step_with_hydrogen(metal_grid, hydrogen_grid, shear_counts_horizontal, shear_counts_vertical, 
                             shear_type, strength_A, strength_B, max_attempts):
    """
    Perform a shear step with probability weighted by the cumulative strength of atoms along the shear line.
    This function enforces that at least one shear step occurs.

    Parameters:
    - metal_grid              : np.array, 2D grid of metal atoms (0 for A, 1 for B).
    - hydrogen_grid           : np.array, 2D grid of hydrogen atoms (1 for H, 0 for empty).
    - shear_counts_horizontal : np.array, tracks horizontal shear steps per row.
    - shear_counts_vertical   : np.array, tracks vertical shear steps per column.
    - shear_type              : int, 0 for horizontal, 1 for vertical, 2 for random selection.
    - strength_A              : float, flow strength (hardness) of metal type A.
    - strength_B              : float, flow strength (hardness) of metal type B.
    - max_attempts            : int, maximum attempts to find a suitable shear location before forcing it.

    Returns:
    - int: The row or column index where shear occurred.
    """

    grid_size_x, grid_size_y = metal_grid.shape

    # **Optimized Shear Type Selection**
    if shear_type == 2:
        horizontal_shear = np.random.rand() < 0.8  # 50% chance of horizontal shear
    else:
        horizontal_shear = shear_type == 0

    # Maximum strength normalization factor
    max_strength = 2 * max(strength_A, strength_B) * (grid_size_y if horizontal_shear else grid_size_x)

    # Attempt up to `max_attempts` times to perform a shear step
    for _ in range(max_attempts):
        if horizontal_shear:
            # Pick a random row
            line = np.random.randint(0, grid_size_x)

            # Compute cumulative strength (current + above row)
            cumulative_strength = np.sum((metal_grid[line, :] == 0) * strength_A + (metal_grid[line, :] == 1) * strength_B) + \
                                  np.sum((metal_grid[(line - 1) % grid_size_x, :] == 0) * strength_A + (metal_grid[(line - 1) % grid_size_x, :] == 1) * strength_B)
                                  
            shear_probability = 1.0 - (cumulative_strength / max_strength)

            # Perform shear step if probability is met or at last attempt
            if np.random.rand() < shear_probability or _ == max_attempts - 1:

                # Efficient row shifting using pure NumPy indexing
                last_col_metal = metal_grid[:line, -1].copy()  # Store last column
                metal_grid[:line, 1:] = metal_grid[:line, :-1]  # Shift right
                metal_grid[:line, 0] = last_col_metal  # Wrap-around

                last_col_hydrogen = hydrogen_grid[:line, -1].copy()  # Store last column
                hydrogen_grid[:line, 1:] = hydrogen_grid[:line, :-1]  # Shift right
                hydrogen_grid[:line, 0] = last_col_hydrogen  # Wrap-around

                shear_counts_horizontal[line] += 1  
                return   # Return immediately after successful shear

        else:
            # Pick a random column
            line = np.random.randint(0, grid_size_y)
            
            # Compute cumulative strength (current + left column)
            cumulative_strength = np.sum((metal_grid[:, line] == 0) * strength_A + (metal_grid[:, line] == 1) * strength_B) + \
                                  np.sum((metal_grid[:, (line - 1) % grid_size_y] == 0) * strength_A + (metal_grid[:, (line - 1) % grid_size_y] == 1) * strength_B)
                                  
            shear_probability = 1.0 - (cumulative_strength / max_strength)

            # Perform shear step if probability is met or at last attempt
            if np.random.rand() < shear_probability or _ == max_attempts - 1:
                
                # Efficient column shifting using pure NumPy indexing
                last_row_metal = metal_grid[-1, :line].copy()  # Store last row
                metal_grid[1:, :line] = metal_grid[:-1, :line]  # Shift down
                metal_grid[0, :line] = last_row_metal  # Wrap-around

                last_row_hydrogen = hydrogen_grid[-1, :line].copy()  # Store last row
                hydrogen_grid[1:, :line] = hydrogen_grid[:-1, :line]  # Shift down
                hydrogen_grid[0, :line] = last_row_hydrogen  # Wrap-around

                shear_counts_vertical[line] += 1  
                return   # Return immediately after successful shear
            
# Function to calculate interaction energy between metal atom and neighboring hydrogen sites
@njit(fastmath=True, cache=True)  # JIT compile for speed, use fastmath optimizations
def calculate_metal_hydrogen_interaction_energy(x, y, metal_grid, hydrogen_grid, E_AH, E_BH):
    """
    Function to calculate the interaction energy between a metal atom and its neighboring hydrogen sites.
    The function considers the four interstitial hydrogen sites surrounding the metal atom, and accounts 
    for periodic boundary conditions to ensure correct interactions at the grid boundaries.
    
    Parameters:
    x, y:                  int, Coordinates of the metal atom on the metal grid.
    metal_grid:            2D numpy array, grid representing metal atoms (A and B) in the system.
    hydrogen_grid:         2D numpy array, grid representing the presence of hydrogen atoms in the interstitial sites.
    E_AH:                  float, interaction energy between metal A atoms and hydrogen.
    E_BH:                  float, interaction energy between metal B atoms and hydrogen.
    
    Returns:
    energy:                float, total interaction energy between the metal atom and its neighboring hydrogen sites.
    """
    energy = 0
    # Define the four neighboring hydrogen sites around the metal atom
    neighbors = [(x, y), (x, (y-1) % grid_size), ((x-1) % grid_size, y), ((x-1) % grid_size, (y-1) % grid_size)]
    
    metal = metal_grid[x, y]  # Get the metal type (A or B)
    
    # Loop over the neighbors and calculate the interaction energy
    for nx, ny in neighbors:
        hydrogen_site = hydrogen_grid[nx, ny]
        if hydrogen_site == 1:  # Check if hydrogen is present
            if metal == 0:  # Metal A
                energy += E_AH
            else:  # Metal B
                energy += E_BH
    
    return energy

# Function to perform a metal diffusion step considering both metal-metal and metal-hydrogen interactions with periodic boundary conditions
@njit(fastmath=True, cache=True)
def metal_diffusion_step_with_hydrogen(metal_grid, hydrogen_grid, kT, E_AA, E_BB, E_AB, E_AH, E_BH, grid_size):
    """
    Function to perform a single metal diffusion step in the system, considering both metal-metal and 
    metal-hydrogen interactions with periodic boundary conditions. The function randomly selects a metal 
    atom and one of its neighbors, attempts to swap them, and applies the Metropolis criterion to determine 
    if the swap is accepted. Periodic boundary conditions are applied to ensure that atoms near the edges 
    of the grid interact correctly with atoms on the opposite side.
    
    Parameters:
    - metal_grid          : 2D numpy array, grid representing metal atoms (A and B) in the system.
    - hydrogen_grid       : 2D numpy array, grid representing the presence of hydrogen atoms in the interstitial sites.
    - kT                  : float, thermal energy factor (k * T), controlling the acceptance of energetically unfavorable moves.
    - E_AA                : float, interaction energy between A-A atoms.
    - E_BB                : float, interaction energy between B-B atoms.
    - E_AB                : float, interaction energy between A-B atoms.
    - E_AH                : float, interaction energy between metal A atoms and hydrogen.
    - E_BH                : float, interaction energy between metal B atoms and hydrogen.
    - max_attempts        : int, maximum number of iterations until which shear is enforced, avoid while loop for compiling.
    
    Returns:
                        : None, function modifies the metal grid in place by either accepting or rejecting the atom swap.
"""
    # Select a random metal atom
    x, y = np.random.randint(0, grid_size), np.random.randint(0, grid_size)

    # Select a random neighboring site (up, down, left, right)
    direction = np.random.randint(0, 4)
    if direction == 0: nx, ny = (x - 1) % grid_size, y  # Up
    elif direction == 1: nx, ny = (x + 1) % grid_size, y  # Down
    elif direction == 2: nx, ny = x, (y - 1) % grid_size  # Left
    else: nx, ny = x, (y + 1) % grid_size  # Right
    
    # Early exit if atoms are identical (no need to swap)
    if metal_grid[x, y] == metal_grid[nx, ny]:
        return True # jump would be successful and therefore return True

    # Calculate the initial energy (metal-metal and metal-hydrogen interactions) before the swap
    initial_energy = (calculate_metal_metal_interaction_energy(   x,   y, metal_grid, E_AA, E_BB, E_AB) + 
                      calculate_metal_hydrogen_interaction_energy(x,   y, metal_grid, hydrogen_grid, E_AH, E_BH)) + \
                     (calculate_metal_metal_interaction_energy(   nx, ny, metal_grid, E_AA, E_BB, E_AB) + 
                      calculate_metal_hydrogen_interaction_energy(nx, ny, metal_grid, hydrogen_grid, E_AH, E_BH))

    # Swap the metal atoms
    metal_grid[x, y], metal_grid[nx, ny] = metal_grid[nx, ny], metal_grid[x, y]

    # Calculate the final energy (metal-metal and metal-hydrogen interactions) after the swap
    final_energy = (calculate_metal_metal_interaction_energy(   x,   y, metal_grid, E_AA, E_BB, E_AB) + 
                    calculate_metal_hydrogen_interaction_energy(x,   y, metal_grid, hydrogen_grid, E_AH, E_BH)) + \
                   (calculate_metal_metal_interaction_energy(   nx, ny, metal_grid, E_AA, E_BB, E_AB) + 
                    calculate_metal_hydrogen_interaction_energy(nx, ny, metal_grid, hydrogen_grid, E_AH, E_BH))

    # Calculate the energy change
    delta_E = final_energy - initial_energy

    accept = True
    # Apply the Metropolis criterion: if the energy increases, accept with a probability exp(-delta_E / kT)
    if delta_E > 0 and np.exp(-delta_E / kT) < np.random.rand():
        accept = False
        # If the move is rejected, reverse the swap
        metal_grid[x, y], metal_grid[nx, ny] = metal_grid[nx, ny], metal_grid[x, y]
        
    return accept

# Function to calculate interaction energy of hydrogen atoms with surrounding metals, including periodic boundary conditions
@njit(fastmath=True, cache=True)  # JIT compile for speed, use fastmath optimizations
def calculate_hydrogen_metal_interaction_energy(x, y, hydrogen_grid, metal_grid, E_AH, E_BH):
    """
    Function to calculate the interaction energy between a hydrogen atom and its surrounding metal atoms.
    This function assumes periodic boundary conditions, so if the hydrogen atom is near the edge of the grid,
    it still interacts with metal atoms on the opposite side.
    
    Parameters:
    x, y                : int, coordinates of the hydrogen atom on the hydrogen grid.
    hydrogen_grid       : 2D numpy array, grid representing the presence of hydrogen atoms in the interstitial sites.
    metal_grid          : 2D numpy array, grid representing metal atoms (A and B) in the system.
    E_AH                : float, interaction energy between metal A atoms and hydrogen.
    E_BH                : float, interaction energy between metal B atoms and hydrogen.
    
    Returns:
    energy              : float, total interaction energy between the hydrogen atom and its surrounding metal atoms.
    """
    energy = 0
    if hydrogen_grid[x, y] == 1:  # Check if there's a hydrogen atom at (x, y)
        # Get the surrounding metal atoms with periodic boundary conditions,
        # Similar but opposite to the metal-hydrogen interactions, as in reallity the hydrogen grid lies inbetween the metal grid
        metal_1 = metal_grid[x % grid_size, y % grid_size]
        metal_2 = metal_grid[x % grid_size, (y+1) % grid_size]
        metal_3 = metal_grid[(x+1) % grid_size, y % grid_size]
        metal_4 = metal_grid[(x+1) % grid_size, (y+1) % grid_size]
        
        # Calculate interaction energy between hydrogen and surrounding metal atoms
        # Similar to the code calculating the metal-hydrogen interactions,
        # but here we can assume that the hydrogen site is occupied, i.e., 1
        for metal in [metal_1, metal_2, metal_3, metal_4]:
            if metal == 0:  # Metal A
                energy += E_AH
            else:  # Metal B
                energy += E_BH
    return energy

# Function to perform a metal diffusion step considering both metal-hydrogen interactions with periodic boundary conditions
@njit(fastmath=True, cache=True)
def hydrogen_diffusion_step(hydrogen_grid, metal_grid, kT, E_AH, E_BH, grid_size, hydrogen_positions):
    """
    Function to perform a hydrogen diffusion step in the system, considering interaction with neighboring metal atoms
    and periodic boundary conditions. A hydrogen atom is randomly selected and swapped with a neighboring empty site,
    and the Metropolis criterion is applied to decide whether the swap is accepted based on the energy change.
    
    Parameters:
    hydrogen_grid       : 2D numpy array, grid representing the presence of hydrogen atoms in the interstitial sites.
    metal_grid          : 2D numpy array, grid representing metal atoms (A and B) in the system.
    kT                  : float, thermal energy factor (k * T), controlling the acceptance of energetically unfavorable moves.
    E_AH                : float, interaction energy between metal A atoms and hydrogen.
    E_BH                : float, interaction energy between metal B atoms and hydrogen.
    
    Returns:
    None                : The function modifies the hydrogen grid in place by either accepting or rejecting the hydrogen atom swap.
    """
    
    # Select a random hydrogen atom
    idx = np.random.randint(0, len(hydrogen_positions))
    x, y = hydrogen_positions[idx]
    
    # Select a random neighbor (up, down, left, right)
    direction = np.random.randint(0, 4)
    if direction == 0: nx, ny = (x - 1) % grid_size, y  # Up
    elif direction == 1: nx, ny = (x + 1) % grid_size, y  # Down
    elif direction == 2: nx, ny = x, (y - 1) % grid_size  # Left
    else: nx, ny = x, (y + 1) % grid_size  # Right

    # Ensure the chosen site is empty (avoid unnecessary swaps)
    if hydrogen_grid[nx, ny] == 1:
        return True  # No swap possible, return early, jump would be successful and therefore return True

    # Calculate the initial energy (before the swap)
    initial_energy = calculate_hydrogen_metal_interaction_energy(x, y, hydrogen_grid, metal_grid, E_AH, E_BH) 

    # Swap hydrogen atoms
    hydrogen_grid[x, y], hydrogen_grid[nx, ny] = hydrogen_grid[nx, ny], hydrogen_grid[x, y]

    # Calculate the final energy (after the swap)
    final_energy = calculate_hydrogen_metal_interaction_energy(nx, ny, hydrogen_grid, metal_grid, E_AH, E_BH) 

    # Calculate the energy change
    delta_E = final_energy - initial_energy

    accept = True
    # Apply the Metropolis criterion
    if delta_E > 0 and np.exp(-delta_E / kT) < np.random.rand():
        accept = False
        # Revert the hydrogen swap if the move is rejected
        hydrogen_grid[x, y], hydrogen_grid[nx, ny] = hydrogen_grid[nx, ny], hydrogen_grid[x, y]
        
    # Update Hydrogen Position Array
    if accept:
        hydrogen_positions[idx, 0] = nx  # Update x-coordinate
        hydrogen_positions[idx, 1] = ny  # Update y-coordinate

    return accept
                               
# Function that sums up all the interaction energies between hydrogen and metals
def calculate_total_metal_hydrogen_interaction_energy(metal_grid, hydrogen_grid, E_AH, E_BH):
    """
    Vectorized calculation of the total metal-hydrogen interaction energy in the grid.

    Parameters:
    - metal_grid       : np.array, 2D array representing metal atom types (0 for A, 1 for B).
    - hydrogen_grid    : np.array, 2D array representing hydrogen atom presence (1 for hydrogen, 0 for empty).
    - E_AH             : float, interaction energy between metal A atoms and hydrogen.
    - E_BH             : float, interaction energy between metal B atoms and hydrogen.

    Returns:
    - total_energy     : float, total interaction energy between all hydrogen atoms and their surrounding metals.
    """
    # Shift the metal grid to simulate the periodic neighbor interactions
    # Hydrogen interacts with the 4 surrounding metal atoms (periodic boundary conditions are applied):
    metal_neighbor_1 = metal_grid  # top left intersitial neighbor
    metal_neighbor_2 = np.roll(metal_grid, shift= -1, axis=1)  # top right intersitial neighbor
    metal_neighbor_3 = np.roll(metal_grid, shift= -1, axis=0)  # lower left intersitial neighbor
    metal_neighbor_4 = np.roll(np.roll(metal_grid, shift= -1, axis=0), shift= -1, axis=1)  # lower right intersitial neighbor

    interaction_AH = 0
    interaction_BH = 0
    for metal_grid in [metal_neighbor_1, metal_neighbor_2, metal_neighbor_3, metal_neighbor_4]:
        # Interaction for A atoms (A = 0), only where hydrogen is present (hydrogen_grid == 1)
        interaction_AH += (metal_grid == 0) * (hydrogen_grid == 1) * E_AH
        # Interaction for B atoms (B = 1), only where hydrogen is present (hydrogen_grid == 1)
        interaction_BH += (metal_grid == 1) * (hydrogen_grid == 1) * E_BH
    
    total_energy = np.sum(interaction_AH) + np.sum(interaction_BH)

    return total_energy

# Function for grouping shear steps for better visualization of the evolution with MC steps
def dynamic_grouping(shear_step):
    if 1 <= shear_step <= 10:
        return shear_step  # Each step gets its own group
    elif 10 < shear_step <= 100:
        return (shear_step // 10) * 10  # Bins of 10
    elif 100 < shear_step <= 1000:
        return (shear_step // 100) * 100  # Bins of 100
    elif 1000 < shear_step <= 10000:
        return (shear_step // 1000) * 1000  # Bins of 1000
    elif 10000 < shear_step <= 100000:
        return (shear_step // 10000) * 10000  # Bins of 1000
    else:
        return (shear_step // 100000) * 100000  # Bins of 1000

#%% FUNCTIONS - VIDEO GENERATION


def apply_colormap_to_array(array, cmap, norm):
    """
    Apply a colormap to a 2D array and return a 3D array (RGB).
    
    Parameters:
    - array: 2D numpy array (values assumed to be 0 or 1).
    - cmap: Matplotlib colormap.
    - norm: Matplotlib Normalize object for scaling values.
    
    Returns:
    - 3D numpy array (RGB image)
    """
    normed_array = norm(array)
    rgba_img = cmap(normed_array)
    return (rgba_img[:, :, :3] * 255).astype(np.uint8)  # Convert RGBA to RGB and scale to 255

def sort_key(filename):
    """
    Extract the enumeration from the filename for sorting.
    Assumes the enumeration is the last number before '.npy' in the filename.
    
    Parameters:
    - filename: str, name of the file
    
    Returns:
    - int, extracted step number
    """
    match = re.search(r'_step_(\d+)\.npy$', filename)
    return int(match.group(1)) if match else 0

def create_video_from_arrays(image_filenames, output_file, fps=30, cmap=None, scale_factor=10, stats_df=None):
    """
    Create a video from a list of filenames representing 2D NumPy arrays.
    Uses a DataFrame for shear step tracking.
    
    Parameters:
    - image_filenames: List of filenames of the images (.npy files).
    - output_file: Path to the output video file (e.g., 'output.mp4').
    - fps: Frames per second for the video.
    - cmap: Matplotlib colormap for mapping 0/1 values to colors.
    - scale_factor: Integer factor to scale up the resolution.
    - stats_df: Pandas DataFrame with "Step" and "Successful Shear Attempts" columns.
    """
    if not image_filenames or stats_df is None:
        raise ValueError("Image list or stats DataFrame is missing. Cannot create a video.")

    # Sort filenames based on the step number
    image_filenames = sorted(image_filenames, key=sort_key)

    # Convert DataFrame into a dictionary for fast lookup: {MC Step -> Shear Steps}
    shear_steps_dict = dict(zip(stats_df["Step"], stats_df["Successful Shear Attempts"]))

    # Load the first image to get dimensions and normalize values
    first_image = np.load(image_filenames[0])
    norm = Normalize(vmin=0, vmax=1)  # Normalize for binary values (0 or 1)

    # Get the original dimensions and scale them
    height, width = first_image.shape
    scaled_height, scaled_width = height * scale_factor, width * scale_factor

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (scaled_width, scaled_height), True)

    # Font settings for overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * scale_factor / 10  # Adjust based on scale factor
    font_thickness = int(2 * scale_factor / 10)
    text_color = (255, 255, 255)  # White text
    shadow_color = (0, 0, 0)  # Black shadow for contrast
    text_offset_x, text_offset_y = int(10 * scale_factor / 10), int(20 * scale_factor / 10)

    # Process and write each image to the video
    for filename in image_filenames:
        img = np.load(filename)  # Load the image
        mc_step = sort_key(filename)  # Extract MC step number from filename
        successful_shear_steps = shear_steps_dict.get(mc_step, 0)  # Get shear steps from DataFrame
        shear_strain = successful_shear_steps / 50  # Normalize shear strain

        img_colored = apply_colormap_to_array(img, cmap, norm)  # Apply colormap
        img_colored_bgr = cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR
        img_scaled = cv2.resize(img_colored_bgr, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)  # Scale up

        # Define text positions
        text_pos_shear = (scaled_width - 220, text_offset_y)  # Position for shear strain
        text_pos_step = (scaled_width - 220, text_offset_y + 25)  # Position for MC step

        # Draw text shadow (black) for better contrast
        cv2.putText(img_scaled, f"Shear Strain: {shear_strain:.4f}", (text_pos_shear[0] + 1, text_pos_shear[1] + 1),
                    font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)
        cv2.putText(img_scaled, f"MC Step: {mc_step}", (text_pos_step[0] + 1, text_pos_step[1] + 1),
                    font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)

        # Draw text in white
        cv2.putText(img_scaled, f"Shear Strain: {shear_strain:.4f}", text_pos_shear,
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(img_scaled, f"MC Step: {mc_step}", text_pos_step,
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        video_writer.write(img_scaled)

    video_writer.release()
    print(f"Video saved as {output_file}")
    
    
def create_metal_hydrogen_video(metal_filenames, hydrogen_filenames, output_file, fps=30, 
                                cmap_metal=None, cmap_hydrogen=None, scale_factor=10, stats_df=None):
    """
    Create a side-by-side video of metal and hydrogen grids from NumPy array files.

    Parameters:
    - metal_filenames: List of filenames of the metal grid images (.npy files).
    - hydrogen_filenames: List of filenames of the hydrogen grid images (.npy files).
    - output_file: Path to the output video file (e.g., 'output.mp4').
    - fps: Frames per second for the video.
    - cmap_metal: Matplotlib colormap for the metal grid.
    - cmap_hydrogen: Matplotlib colormap for the hydrogen grid.
    - scale_factor: Integer factor to scale up the resolution.
    - stats_df: Pandas DataFrame with "Step" and "Successful Shear Attempts" columns.
    """
    if not metal_filenames or not hydrogen_filenames or stats_df is None:
        raise ValueError("Metal or hydrogen image list or stats DataFrame is missing.")

    # Sort filenames based on the step number
    metal_filenames = sorted(metal_filenames, key=sort_key)
    hydrogen_filenames = sorted(hydrogen_filenames, key=sort_key)

    # Convert DataFrame into a dictionary for fast lookup: {MC Step -> Shear Steps}
    shear_steps_dict = dict(zip(stats_df["Step"], stats_df["Successful Shear Attempts"]))

    # Load the first image to get dimensions and normalize values
    first_metal = np.load(metal_filenames[0])
    first_hydrogen = np.load(hydrogen_filenames[0])

    norm_metal = Normalize(vmin=0, vmax=1)  
    norm_hydrogen = Normalize(vmin=0, vmax=1)  

    # Get the original dimensions and scale them
    height, width = first_metal.shape
    scaled_height, scaled_width = height * scale_factor, width * scale_factor

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for MP4
    video_writer = cv2.VideoWriter(output_file, fourcc, fps, (2 * scaled_width, scaled_height), True)

    # Font settings for overlay text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5 * scale_factor / 10  
    font_thickness = int(2 * scale_factor / 10)
    text_color = (255, 255, 255)  
    shadow_color = (0, 0, 0)  
    text_offset_x, text_offset_y = int(10 * scale_factor / 10), int(20 * scale_factor / 10)

    # Process and write each image to the video
    for metal_file, hydrogen_file in zip(metal_filenames, hydrogen_filenames):
        metal_img = np.load(metal_file)  
        hydrogen_img = np.load(hydrogen_file)  
        
        mc_step = sort_key(metal_file)  
        successful_shear_steps = shear_steps_dict.get(mc_step, 0)  
        shear_strain = successful_shear_steps / 50  

        metal_colored = apply_colormap_to_array(metal_img, cmap_metal, norm_metal)  
        hydrogen_colored = apply_colormap_to_array(hydrogen_img, cmap_hydrogen, norm_hydrogen)  

        metal_bgr = cv2.cvtColor(metal_colored, cv2.COLOR_RGB2BGR)  
        hydrogen_bgr = cv2.cvtColor(hydrogen_colored, cv2.COLOR_RGB2BGR)  

        metal_scaled = cv2.resize(metal_bgr, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)
        hydrogen_scaled = cv2.resize(hydrogen_bgr, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

        # **Create side-by-side frame**
        combined_frame = np.hstack((metal_scaled, hydrogen_scaled))

        # Define text positions
        text_pos_shear = (2 * scaled_width - 220, text_offset_y)
        text_pos_step = (2 * scaled_width - 220, text_offset_y + 25)

        # Draw text shadow (black) for contrast
        cv2.putText(combined_frame, f"Shear Strain: {shear_strain:.4f}", (text_pos_shear[0] + 1, text_pos_shear[1] + 1),
                    font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)
        cv2.putText(combined_frame, f"MC Step: {mc_step}", (text_pos_step[0] + 1, text_pos_step[1] + 1),
                    font, font_scale, shadow_color, font_thickness, cv2.LINE_AA)

        # Draw text in white
        cv2.putText(combined_frame, f"Shear Strain: {shear_strain:.4f}", text_pos_shear,
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)
        cv2.putText(combined_frame, f"MC Step: {mc_step}", text_pos_step,
                    font, font_scale, text_color, font_thickness, cv2.LINE_AA)

        video_writer.write(combined_frame)

    video_writer.release()
    print(f"Video saved as {output_file}")



#%% METAL-SYSTEM - DEFINITIONS (REQUIRED for Simulation and Plotting)
# Required for all plotting and the MC simulation itself.
# Definitions for plotting - fontsizes & colors

# Create a custom color map similar to the blue and red tones from the Seaborn "deep" palette
colors = sns.color_palette("deep", 4)
colors = colors[0], colors[3]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Define the font size
fs = 10


################################################################################################
# Definitions of grid size, number of simulationm and the convergence criteria
# Grid size
grid_size = 50
# Binary (thresholded) SEM image with Real starting configuration
starting_conf = "Real_starting_conf.png"

# Define the (maximum) number of Monte Carlo steps per simulation and number of repetitions
n_mc_steps = 100000000
n_simulations = 3

# Define convergence criteria for the MC simulation
# Parameters for stopping criterion
n_persistence =  10         # Number of consecutive stable windows
threshold_mean = 0.005       # Threshold for change in mean of mixing index


################################################################################################
# Definiations of shear probabilities, interaction energies and the "temperature" of the system (300K)

# Shear probabilities for every MC step - the higher the diffusion coefficient the lower the probability for shear
# 50x50 Grid - HPT deformation speed of 1.27 rpm - For different diffusion coefficients
p_shear_list = [0.00000045, # 10-15 m²/s
                0.00000447, # 10-16 m²/s
                0.00004474, # 10-17 m²/s
                0.00044719, # 10-18 m²/s
                0.00445399, # 10-19 m²/s
                0.04282333, # 10-20 m²/s
                0.30910222, # 10-21 m²/s
                0.81731560, # 10-22 m²/s
                0.97813692, # 10-23 m²/s
                0.99776981, # 10-24 m²/s
                0.99977653, # 10-25 m²/s
                0.99997765] # 10-26 m²/s


simulation_inputs = {
                    "Diffusion coefficient"   : [1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23, 1e-24, 1e-25, 1e-26],
                    "Shear probability"       : p_shear_list,
                    "Diffusion probability"   : 1 - np.array(p_shear_list)
                   }

simulation_inputs = pd.DataFrame(simulation_inputs)

# Energy parameters and assocaited (assumed) standard deviations - Interaction energies between A-A, B-B and A-B atoms 
# Factor 1000 - kJ to J; Factor 1/4 - Complete mixing ethalpy only obtained when bonding with e.g. 4 different atoms, i.e., 4 A-B pairs,
## Factor 2 due to the douple counting issue 
E_AA_mean = 2 * (1000/4) *  2.66E-25    # Enthalpy of formation TiVZrNbHf                               - kJ atom-1      
E_BB = 0                 # Enthalpy of formation Cu - Defined as zero                                   - kJ atom-1 
E_AB_mean = 2 * (1000/4) * -1.21E-23    # Enthalpy of formation TiVZrNbHf-Cu - Real atomic composition  - kJ atom-1     

Estd = False
E_AA_std = abs(E_AA_mean) * Estd
E_AB_std = abs(E_AB_mean) * Estd

# Defining kT for HPT conditions close to RT (300K)
kT = constants.Boltzmann * 300

#%% METAL-SYSTEM - MONTE CARLO SIMULATION WITH VARYING DIFFUSION COEFFICIENTS
########################### ACTUAL SIMULATION ################################

# LOOP 1 - Iterate through the shear probabilities associated with the different diffusion coefficients
#          i.e., Results as a FUNCTION OF DIFFUSION COEFFICIENT 
for index, simulation_input in simulation_inputs.iterrows():
    
    p_shear           = simulation_input["Shear probability"]
    D                 = simulation_input["Diffusion coefficient"]
    p_metal_diffusion = simulation_input["Diffusion probability"]
    
    # Create a folder where to save all the generated results
    folder_name = f"D_{D}_MC_{n_mc_steps}_steps_{n_simulations}_runs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    ##############################################################################################################
    # Initializing all the DataFrames, Lists and other containers for saving all the relevant information of the
    # MC simulation for evaluation and plotting.
    
    # Track the simulation results for every run and step.
    # Steps are sampled according to "get_sampling_interval" - Otherwise too much data would be generated.
    
    # DataFrame for saving all the step data - All runs, all sampled steps
    all_simulation_data = pd.DataFrame()
    
    # For saving the final grids in this list for plotting latter the results plots. (e.g., simulation_results_run0.png)
    metal_grids = []
    all_simulation_data = []
    
    # Initial all lists to calculate the shear and diffusion rates etc.
    # This is ONLY for calculating the individual probabilities and printing them latter in the console - not important for the evaluation latter.
    metal_diffusion_attempt_rates = []
    metal_diffusion_success_rates = []
    shear_success_rates = []
    
    # This is for saving all the final grid stats and shear / diffusion attempts at the end of EACH RUN.
    final_stats = []
    
    # List to track when convergence has been achieved
    convergence_step = []
    ##############################################################################################################
    
    
    # LOOP 2 - Iterate through the MC runs, each being a unique complete MC simulation with its own samples 
    #          interaction energies; however, all the runs have the same shear and diffusion probabilies determined
    #          by LOOP 1
    for run in tqdm(range(n_simulations), desc="Simulations Progress"):
        
        #########################################################################################################
        # Sample the interaction energies and save them for later evaluation
        
        # Sample the energies associated A-A and A-B neighbors from a normal distribution. 
        if Estd:
            E_AA = sample_energy(E_AA_mean, E_AA_std)
            E_AB = sample_energy(E_AB_mean, E_AB_std)
        # Or just that a single input value.
        else:
            E_AA = E_AA_mean
            E_AB = E_AB_mean
        #########################################################################################################
        
        
        #########################################################################################################
        # Initializing all the DataFrames, Lists and other containers for saving all the relevant information of the
        # MC simulation for evaluation and plotting.
        
        # Initialize variables for counting successfull diffusion and shear steps for every MC run
        successful_metal_diffusion_attempts = 0
        successful_shear_attempts = 0
        total_metal_diffusion_attempts = 0
        
        # Grid for tracking the shear localization implementet
        shear_counts_horizontal = np.zeros(grid_size, dtype=np.int64)
        shear_counts_vertical   = np.zeros(grid_size, dtype=np.int64)
        metal_grid = initialize_grid_from_image(starting_conf, grid_size)
        

        # Initiate the convergence criteria for stopping the MC simulation
        converged = False  # Flag to indicate convergence - Reset for before every MC run (i.e., for LOOP 3)
        convergence_counter = 0 # Counter for steps fullfilling the convergence - Reset for before every MC run (i.e., for LOOP 3)
        #########################################################################################################
         
        # Use NumPy’s random function for efficiency
        rng = np.random.default_rng()
        
        sampling_interval = 5
        
        total_energy_list = []
        mixing_index_list = []
        
        # LOOP 3 - Iterate through the individual MC steps for the MC run defined in LOOP 2
        for step in range(n_mc_steps):
            
            random_value = rng.random(1)
            
            # Attempt a diffusion step
            if random_value < p_metal_diffusion:
                # Count all diffusion attempts
                total_metal_diffusion_attempts += 1
                # Make an MC diffusion step based on the interaction energy and the kT
                if metal_diffusion_step(metal_grid, kT=kT, E_AA=E_AA, E_BB=E_BB, E_AB=E_AB, grid_size=grid_size):
                    successful_metal_diffusion_attempts += 1

            # Attempt a shear step
            if random_value < p_shear:
                # Count all shear attempts
                successful_shear_attempts += 1
                # Shear attempts are always successfull
                # shear_step(metal_grid, shear_type="horizontal", shear_counts_horizontal=shear_counts)
                shear_step(metal_grid,
                           shear_type=2,
                           strength_A=1,
                           strength_B=0.2,
                           shear_counts_horizontal=shear_counts_horizontal,
                           shear_counts_vertical=shear_counts_vertical,
                           max_attempts=100)

            
            # If the MC step is a multiple of the sampling interval then the energy and mixing index is calulated and saved
            if step % sampling_interval == 0:
                # Get the sampling interval for the MC step
                sampling_interval = get_sampling_interval(step)
                
                total_energy = calculate_total_metal_metal_interaction_energy(metal_grid, E_AA, E_BB, E_AB)
                mixing_index = calculate_mixing_index(metal_grid)
                total_energy_list.append(total_energy)
                mixing_index_list.append(mixing_index)
            
                video_folder = f"{folder_name}/for_video/run_{run}"
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                np.save(f"{folder_name}/for_video/run_{run}/metal_grid_run_{run}_step_{step}", metal_grid)
     
                
                all_simulation_data.append([run, step, total_energy, mixing_index,
                                            successful_shear_attempts,
                                            total_metal_diffusion_attempts, successful_metal_diffusion_attempts])
                
                #################################################################################################
                # CHECK CONVERGENCE CRITERIA
                # Implementation of convergence criteria
                
                # Check for convergence every `n_window` steps
                if not converged and step > 100: #and step % sampling_interval == 0 :
                    # Calc the mean of the last 10 simulation results (to reduce effect of random variation)
                    _current = np.mean(mixing_index_list[-5:])
                    std_current = np.std(mixing_index_list[-5:])
                    
                    # Calc the mean of the simulation results a little further away
                    _previous = np.mean(mixing_index_list[-20:-15])
                    
                    # Calc the changes in the mixing index for checking convergence
                    diff = abs(_current - _previous)
                    
                    # Save the values calculated for the convergence criteria for plotting / evaluating the convergence 
                    all_simulation_data[-1].append(diff)
                    all_simulation_data[-1].append(std_current)
                    
                    if step > 1000000:
                        # CHECK FOR CONVERGENCE - IF CONVERGED -> ADD TO THE CONVERGENCE COUNTER
                        if diff < threshold_mean:
                            convergence_counter += 1
          
                        # IF NOT CONVERGED - RESET THE CONVERGENCE COUNTER TO ZERO
                        else:
                            convergence_counter = 0  # Reset if stability is broken
                    
                        # Stop the simulation if stable for `n_persistence` windows
                        if convergence_counter >= n_persistence:
                            print(f"Convergence reached at step {step}")
                            converged = True
                            continuation_counter = np.ceil(step/10)
                            convergence_step.append(step)
                        
                #################################################################################################
                
                
            
            #####################################################################################################
            # ENFORCE CONVERGENCE - after the simulation has convergened and the coninuation counter is reached
            
            # Handle continuation phase - Continue MC simualtion for `continuation_counter` amount of steps
            if converged:
                if continuation_counter > 0:
                    continuation_counter -= 1

                # STOP SIMULATION
                else:
                    print(f"Continuation phase completed. Stopping simulation at step {step}")
                    break
            #####################################################################################################
            
        # If no converence is reached use the last MC step as the `convergence step`,
        # i.e. the step at which convergence was reached
        if not converged:
            convergence_step.append(all_simulation_data[-1][1])
            
        #########################################################################################################
        # Results post processing after each MC run
        
        # After each run, calculate the attempt and success rates of both diffusion and shear
        # Diffusion attemps
        fraction_diffusion_steps = total_metal_diffusion_attempts / step
        # Successful diffusion attemps - passed the Metropolis criterion
        fraction_successful_diffusion_steps = successful_metal_diffusion_attempts / step
        # Shear attemps = Successful shear attempt - Shear not goverend by interaction energies and thermodynamics
        fraction_shear_steps = successful_shear_attempts / step
    
        # Save the calculated attempt and success rates for each run
        metal_diffusion_attempt_rates.append(fraction_diffusion_steps)
        metal_diffusion_success_rates.append(fraction_successful_diffusion_steps)
        shear_success_rates.append(fraction_shear_steps)
    
        # Calculate mean and std for the recorded values of total energy and the mixing index in the persistence window 
        # Values should have converged already in this window - Save these final stats for each run
        mean_total_energy = np.mean(total_energy_list[-n_persistence:])
        std_total_energy  = np.std( total_energy_list[-n_persistence:])
        mean_mixing_index = np.mean(mixing_index_list[-n_persistence:])
        std_mixing_index  = np.std( mixing_index_list[-n_persistence:])

        # Save all the final stats of the 
        final_stats.append({
            'Run'                           : run,                                   # Which MC run was this
            'Metal Diffusion Probability'   : p_metal_diffusion,                     # Nominal diffusion probability
            'Shear Probability'             : p_shear,                               # Nominal shear probability
            'E_AA'                          : E_AA,                                  # A-A interaction energy during this run
            'E_BB'                          : E_BB,                                  # B-B interaction energy during this run
            'E_AB'                          : E_AB,                                  # A-B interaction energy during this run
            'Mean Total Energy'             : mean_total_energy,                     # Mean energy at the end of the simulation
            'Std Total Energy'              : std_total_energy,                      # Std of the energy at the end of the simulation
            'Mean Mixing Index'             : mean_mixing_index,                     # Mean mixing index at the end of the simulation
            'Std Mixing Index'              : std_mixing_index,                      # Std of the mixing index at the end of the simulation
            "Steps"                         : step,                                  # Number performed MC steps
            "Diffusion Attemps"             : total_metal_diffusion_attempts,        # Diffusion attemps
            "Successful Diffusion Attempts" : successful_metal_diffusion_attempts,   # Successful diffusion attemps
            "Successful Shear Attempts"     : successful_shear_attempts,             # Successful shear attemps
            "Shear Hist Horizontal"         : shear_counts_horizontal,               # Save Numpy array saving where shear steps were made
            "Shear Hist Vertical"           : shear_counts_horizontal,               # Save Numpy array saving where shear steps were made
        })
        
      
        # Save the final metal_grid as numpy array for further evaluation and plotting 
        np.save(f"{folder_name}/metal_grid_run_{run}", metal_grid)
        metal_grids.append(metal_grid)
        
        #########################################################################################################
        
    # Calculate mean and standard deviation for success rates
    mean_diffusion_attempt_rate = np.mean(metal_diffusion_attempt_rates)
    std_diffusion_attempt_rate = np.std(metal_diffusion_attempt_rates)
    mean_diffusion_success_rate = np.mean(metal_diffusion_success_rates)
    std_diffusion_success_rate = np.std(metal_diffusion_success_rates)
    mean_shear_rate = np.mean(shear_success_rates)
    std_shear_rate = np.std(shear_success_rates)

    # Print the mean and std of success rates
    print(f"Diffusion coefficient : {D:.1e}")
    print(f"Mean diffusion attempt rate: {mean_diffusion_attempt_rate:.4f} ± {std_diffusion_attempt_rate:.4f}")
    print(f"Mean diffusion success rate: {mean_diffusion_success_rate:.4f} ± {std_diffusion_success_rate:.4f}")
    print(f"Mean shear success rate:     {mean_shear_rate:.4f}             ± {std_shear_rate:.4f}")

    # Save all the simulation data collected during the simulation and the final simulation stats
    all_simulation_data = pd.DataFrame(all_simulation_data, columns=['Run',
                                                                     'Step',
                                                                     'Total Energy',
                                                                     'Mixing Index', 
                                                                     'Successful Shear Attempts',
                                                                     'Total Metal Diffusion Attempts',
                                                                     'Successful Metal Diffusion Attempts',
                                                                     'Mixing Index Difference',
                                                                     'Mixing Index Std'])
    all_simulation_data.to_csv(f'{folder_name}/simulation_results.csv', index=False)
    
    final_stats_df = pd.DataFrame(final_stats)
    final_stats_df.to_csv(f'{folder_name}/final_stats.csv', index=False)
    
    ##############################################################################################################
    # Plotting all results in a single masterplot for every MC run - All parameters are plotted as a fucntion of MC steps
    for i, c in enumerate(np.array(convergence_step)):
        simulation_data = all_simulation_data.loc[all_simulation_data["Run"] == i]
        metal_grid = metal_grids[i]
        fig, ax = plt.subplots(3,2, figsize=(11, 12), facecolor='white')
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.75, top=0.9)

        colors = sns.color_palette("deep", 4)    

        ax[0,0].imshow(metal_grid, cmap=cmap, origin='upper')
        ax[0,0].set_xlabel("$x$ / unitless")
        ax[0,0].set_ylabel("$y$ / unitless")

        ax[1,0].semilogx(simulation_data["Step"], simulation_data["Total Energy"], color = colors[0])
        ax[1,0].semilogx(simulation_data["Step"].iloc[-1], simulation_data["Total Energy"].iloc[-1],
                        marker="o", markersize=6, color="black")
        ax[1,0].set_ylabel("$Total$ $Energy$ / arb.u.")
        ax[1,0].set_box_aspect(1)

        ax[2,0].semilogx(simulation_data["Step"], simulation_data["Mixing Index"], color = colors[0])
        ax[2,0].semilogx(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index"].iloc[-1],
                        marker="o", markersize=6, color="black")
        ax[2,0].set_xlabel("$Monte$ $Carlos$ $Step$ / unitless")
        ax[2,0].set_ylabel("$Mixing$ $Index$ / unitless")
        ax[2,0].set_box_aspect(1)

        ax[0,1].semilogx(simulation_data["Step"], simulation_data["Mixing Index Difference"], color = colors[0], label="Mean")
        ax[0,1].semilogx(simulation_data["Step"], simulation_data["Mixing Index Std"], color = colors[1], label="Std")
        ax[0,1].semilogx(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index Difference"].iloc[-1],
                         marker="o", markersize=6, color="black")
        ax[0,1].axhline(threshold_mean, linestyle="--", color="black")
        ax[0,1].set_ylabel("$Mixing$ $Index$ / unitless")
        ax[0,1].set_box_aspect(1)
        ax[0,1].legend(loc="upper left", bbox_to_anchor=[1.05, 1])

        ax[1,1].semilogx(simulation_data["Step"], simulation_data["Successful Shear Attempts"], color = colors[0], label="Successful shear attempts")
        ax[1,1].semilogx(simulation_data["Step"], simulation_data["Total Metal Diffusion Attempts"], color = colors[1], label="Diffusion attempts", linestyle="--")
        ax[1,1].semilogx(simulation_data["Step"], simulation_data["Successful Metal Diffusion Attempts"], color = colors[2], label="Successful diffusion attempts")
        ax[1,1].set_ylabel("$Attemps$ / unitless")
        ax[1,1].set_box_aspect(1)
        ax[1,1].legend(loc="upper left", bbox_to_anchor=[1.05, 1])

        ax[2,1].semilogx(simulation_data["Step"], simulation_data["Successful Shear Attempts"]                 / (simulation_data["Step"]+1), color = colors[0], label="Successful shear attempts") 
        ax[2,1].semilogx(simulation_data["Step"], simulation_data["Total Metal Diffusion Attempts"]    / (simulation_data["Step"]+1), color = colors[1], label="Diffusion attempts", linestyle="--")
        ax[2,1].semilogx(simulation_data["Step"], simulation_data["Successful Metal Diffusion Attempts"]  / (simulation_data["Step"]+1), color = colors[2], label="Successful diffusion attempts")      
        ax[2,1].set_xlabel("$Monte$ $Carlos$ $Step$ / unitless")
        ax[2,1].set_ylabel("$Attempt$ $ratios$ / unitless")
        ax[2,1].set_box_aspect(1)
        ax[2,1].legend(loc="upper left", bbox_to_anchor=[1.05, 1])

        fig.suptitle(f'Monte Carlo Run {i} - D={D} m$^{2}$/s', fontsize=12)

        plt.savefig(f'{folder_name}/simulation_results_run_{i}.png', dpi=300)
    ##############################################################################################################
    
    
    

    ##############################################################################################################
    # LINEAR PLOT - Plot results all MC runs for this diffusion coefficient
    fig, ax = plt.subplots(2, 1, figsize=(8.5*cm,  15*cm))
    fig.subplots_adjust(left=0.2, right=0.75)
    
    colors = sns.color_palette("viridis", n_simulations)
    
    # Plot the energy and mixing index for every MC run as a function of MC steps - mark the convergence point with a "o"
    for i, c in enumerate(np.array(convergence_step)):
        simulation_data = all_simulation_data.loc[all_simulation_data["Run"] == i]
        color = colors[i]
        ax[0].plot(simulation_data["Step"], simulation_data["Total Energy"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].plot(simulation_data["Step"], simulation_data["Mixing Index"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].plot(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index"].iloc[-1],
                   label=f'Run {i}', marker="o", markersize=6, color=color)
        
    # Axis labels and color bar
    ax[0].set_ylabel('$Total$ $Energy$ / arb. u.', size=fs)
    ax[1].set_xlabel('$Monte$ $Carlo$ $step$ / unitless', size=fs)
    ax[1].set_ylabel('$Mixing$ $Index$ / unitless', size=fs)
    
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    
    # Add a colorbar related to the MC runs
    cax = fig.add_axes([ax[1].get_position().x1+0.05,ax[1].get_position().y0,0.05,ax[0].get_position().y1-ax[1].get_position().y0])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_simulations - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", pad=0.05)
    cbar.set_label('Run Number', size=fs)
    
    # Save the plot
    plt.savefig(f'{folder_name}/simulation_results_linear.png', dpi=300)
    ##############################################################################################################
    
    
    
    ##############################################################################################################
    # LOG PLOT - Plot results all MC runs for this diffusion coefficient
    fig, ax = plt.subplots(2, 1, figsize=(8.5*cm,  15*cm))
    fig.subplots_adjust(left=0.2, right=0.75)
    
    colors = sns.color_palette("viridis", n_simulations)
     
    # Plot the energy and mixing index for every MC run as a function of MC steps - mark the convergence point with a "o"
    for i, c in enumerate(np.array(convergence_step)):
        simulation_data = all_simulation_data.loc[all_simulation_data["Run"] == i]
        color = colors[i]
        ax[0].semilogx(simulation_data["Step"], simulation_data["Total Energy"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].semilogx(simulation_data["Step"], simulation_data["Mixing Index"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].semilogx(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index"].iloc[-1],
                   label=f'Run {i}', marker="o", markersize=6, color=color)
        
    # Axis labels and color bar
    ax[0].set_ylabel('$Total$ $Energy$ / arb. u.', size=fs)
    ax[1].set_xlabel('$Monte$ $Carlo$ $step$ / unitless', size=fs)
    ax[1].set_ylabel('$Mixing$ $Index$ / unitless', size=fs)
    
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    
    # Add a colorbar related to the MC runs
    cax = fig.add_axes([ax[1].get_position().x1+0.05,ax[1].get_position().y0,0.05,ax[0].get_position().y1-ax[1].get_position().y0])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_simulations - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", pad=0.05)
    cbar.set_label('Run Number', size=fs)
    
    # Save the plot
    plt.savefig(f'{folder_name}/simulation_results_log.png', dpi=300)
    
    plt.close("all")
    ##############################################################################################################
    
#%% METAL-SYSTEM PLOTTING - MIXING INDEX & TOTAL ENERGY vs DIFFUSION COEFFICIENT

FOLDER = "Example_metal_system"

# Gather CSV files
files = glob(FOLDER+"/*/final_stats.csv")
# Reorganize the order of files, ensuring correct concatenation


# Set up colormap
cmap = sns.color_palette("viridis", as_cmap=True)

# Create subplots
fig, ax = plt.subplots(2, 1, figsize=(5, 10), sharex=True)

# Iterate through shear probabilities, diffusion coefficients, and file paths
for i, (p_shear, D, file) in enumerate(zip(simulation_inputs["Shear probability"], simulation_inputs["Diffusion coefficient"], files)):
    
    # Read data from the current CSV file
    stats = pd.read_csv(file)
    
    # Map color based on the mean of "Successful Shear Attempts" using a logarithmic scale
    color = cmap(mcolors.LogNorm(vmin=0.1, vmax=8000 * 100)(stats["Successful Shear Attempts"].mean()))
    
    # Plot the mean and standard deviation of the Mixing Index
    ax[0].errorbar(
        D,
        stats["Mean Mixing Index"].mean(),
        yerr=stats["Mean Mixing Index"].std(),
        marker="o",
        linestyle="-",
        capsize=3,
        capthick=2,
        color=color
    )

# Second loop to plot the Total Energy
for i, (p_shear, D, file) in enumerate(zip(simulation_inputs["Shear probability"], simulation_inputs["Diffusion coefficient"], files)):
    
    # Read data from the current CSV file
    stats = pd.read_csv(file)
    
    # Map color based on the mean of "Successful Shear Attempt" using a logarithmic scale
    color = cmap(mcolors.LogNorm(vmin=0.1, vmax=8000 * 100)(stats["Successful Shear Attempts"].mean()))
    
    # Plot the mean and standard deviation of the Total Energy, scaled by Avogadro's constant
    ax[1].errorbar(
        D,
        (constants.Avogadro * stats["Mean Total Energy"].mean()) / (1000 * 50**2),  # Convert to kJ/mol for a 50x50 metal_grid
        yerr=(constants.Avogadro * stats["Mean Total Energy"].std()) / (1000 * 50**2),
        marker="o",
        linestyle="-",
        capsize=3,
        capthick=2,
        color=color
    )

# Configure the first subplot (Mixing Index)
ax[0].set_xscale('log')  # Set x-axis to logarithmic scale
ax[1].set_xlabel("$D$ / m$^{2}$ s$^{-1}$")  # Label for x-axis
ax[0].set_ylabel("$Mean$ $Mixing$ $Index$ / unitless")  # Label for y-axis
ax[0].set_xlim((10**-30, 10**-12))  # Set x-axis limits
ax[0].set_ylim((0, 1))  # Set y-axis limits

# Configure the second subplot (Total Energy)
ax[1].set_ylabel("\u0394$H_{mix}$ / kJ mol$^{\u22121}$")  # Label for y-axis
ax[1].set_ylim((-10, 0))  # Set y-axis limits

# Create a logarithmic colorbar for the number of Successful Shear Attempts
sm = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.LogNorm(vmin=0.1, vmax=8000 * 100))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.05)
cbar.set_label('$Successful$ $Shear$ $Attempt$ / unitless', size=12)

# Save the figure
plt.savefig(FOLDER+"/MixingIndex_Energy_vs_DiffusionCoeff.png", dpi=300)


#%% METAL-SYSTEM PLOTTING - MIXING INDEX & TOTAL ENERGY vs MC STEPS & STRAIN

FOLDER = "Example_metal_system"

# Define font size and figure size
fs = 9
fig, ax = plt.subplots(2, 2, figsize=(17.2*cm, 15.0*cm), sharex="col")
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95)

# Define the file path pattern for all files you want to include
file_pattern = FOLDER+"/*/simulation_results.csv"

# Get list of result and stats files
result_files = sorted(glob(file_pattern))

# Set up color map
colors = sns.color_palette("viridis", len(result_files))

# Loop through each file
for j, (result_file, D) in enumerate(zip(result_files, simulation_inputs["Diffusion coefficient"])):
    
    # Load the data
    results_data = pd.read_csv(result_file)
    
    # Group by 'Step' to calculate mean and std for 'MixingIndex' and 'Energy'
    result_mean_std = results_data.groupby('Step').agg({
        'Mixing Index': ['mean', 'std'],
        'Total Energy': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns for cleaner column names
    result_mean_std.columns = ['Step', 'MixingIndex_mean', 'MixingIndex_std', 'TotalEnergy_mean', 'TotalEnergy_std']

    result_mean_std["TotalEnergy_mean_molar"] = (constants.Avogadro * result_mean_std["TotalEnergy_mean"]) / (1000 * 50**2)
    result_mean_std["TotalEnergy_std_molar"] = (constants.Avogadro * result_mean_std["TotalEnergy_std"]) / (1000 * 50**2)
    
    # Plot with color based on file index
    color = colors[j]
    ax[0,0].plot(result_mean_std["Step"],
                   result_mean_std["TotalEnergy_mean_molar"],
                   color=color,
                   alpha=1,
                   marker="",
                   linestyle="-",
                   label=str(D))
    
    ax[0,0].fill_between(result_mean_std["Step"],
                       result_mean_std["TotalEnergy_mean_molar"] - result_mean_std["TotalEnergy_std_molar"],
                       result_mean_std["TotalEnergy_mean_molar"] + result_mean_std["TotalEnergy_std_molar"],
                       color=color,
                       alpha=0.2)
    
    ax[0,0].set_xscale('log')
    
    ax[1,0].plot(result_mean_std["Step"],
                   result_mean_std["MixingIndex_mean"],
                   color=color,
                   alpha=1,
                   marker="",
                   linestyle="-",
                   label=str(D))
    
    ax[1,0].fill_between(result_mean_std["Step"],
                       result_mean_std["MixingIndex_mean"] - result_mean_std["MixingIndex_std"],
                       result_mean_std["MixingIndex_mean"] + result_mean_std["MixingIndex_std"],
                       color=color,
                       alpha=0.2)
    ax[1,0].set_xscale('log')
    
    
    results_data['Successful Shear Attempts Grouped'] = results_data['Successful Shear Attempts'].apply(dynamic_grouping)   
    
    # Group by 'Step' to calculate mean and std for 'MixingIndex' and 'Energy'
    result_straingrouped = results_data.groupby('Successful Shear Attempts Grouped').agg({
        'Mixing Index': ['mean', 'std'],
        'Total Energy': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns for cleaner column names
    result_straingrouped.columns = ['Successful Shear Attempts Grouped', 'MixingIndex_mean', 'MixingIndex_std', 'TotalEnergy_mean', 'TotalEnergy_std']

    # Convert the Total Energy to the Total Molar Energy i.e. kJ mol-1
    result_straingrouped["TotalEnergy_mean_molar"] = (constants.Avogadro * result_straingrouped["TotalEnergy_mean"]) / (1000 * 50**2)
    result_straingrouped["TotalEnergy_std_molar"]  = (constants.Avogadro * result_straingrouped["TotalEnergy_std"]) / (1000 * 50**2)
    
    # Plot with color based on file index
    # Divide number of Successful Shear Attempts by 50 to get the actuall shear strain - 50 Successful Shear Attempts are a Shear Strain of 1.
    color = colors[j]
    ax[0,1].plot(result_straingrouped["Successful Shear Attempts Grouped"]/50,
               result_straingrouped["TotalEnergy_mean_molar"],
               color=color,
               alpha=1,
               marker="",
               linestyle="-",
               label=str(D))
    
    ax[0,1].fill_between(result_straingrouped["Successful Shear Attempts Grouped"]/50,
                       result_straingrouped["TotalEnergy_mean_molar"] - result_straingrouped["TotalEnergy_std_molar"],
                       result_straingrouped["TotalEnergy_mean_molar"] + result_straingrouped["TotalEnergy_std_molar"],
                       color=color,
                       alpha=0.2)
    
    ax[0,1].set_xscale('log')
    
    ax[1,1].plot(result_straingrouped["Successful Shear Attempts Grouped"]/50,
               result_straingrouped["MixingIndex_mean"],
               color=color,
               alpha=1,
               marker="",
               linestyle="-",
               label=str(D))
    
    ax[1,1].fill_between(result_straingrouped["Successful Shear Attempts Grouped"]/50,
                       result_straingrouped["MixingIndex_mean"] - result_straingrouped["MixingIndex_std"],
                       result_straingrouped["MixingIndex_mean"] + result_straingrouped["MixingIndex_std"],
                       color=color,
                       alpha=0.2)
    ax[1,1].set_xscale('log')

# Set titles, labels, and legends
#ax[0,0].set_xlabel('Strain / unitless', size=fs)
ax[0,0].set_ylabel('\u0394$H_{mix}$ / kJ mol$^{\u22121}$', size=fs)
ax[1,0].set_xlabel('$Monte$ $Carlo$ $Step$ / unitless', size=fs)
ax[1,0].set_ylabel('$Mixing$ $Index$ / unitless', size=fs)

#ax[0,0].set_xlim(adjust_margins((10**1, 10**8), ax[0,0], scaling="log", percent=0.015))

ax[0,0].set_box_aspect(1)
ax[1,0].set_box_aspect(1)

ax[1,0].set_ylim((0,1))

# Set titles, labels, and legends
ax[1,1].set_xlabel('$Shear$ $Strain$ $\u03B3$ / unitless', size=fs)

ax[0,1].set_xlim(adjust_margins((10**1, 10**8), ax[0,1], scaling="log", percent=0.015))
ax[1,1].set_xlim(adjust_margins((10**-1, 10**6), ax[1,1], scaling="log", percent=0.015))

ax[0,0].set_ylim(adjust_margins((-10, 0), ax[0,0], percent=0.015))
ax[0,1].set_ylim(adjust_margins((-10, 0), ax[0,1], percent=0.015))
ax[1,0].set_ylim(adjust_margins((0, 1), ax[1,0], percent=0.015))
ax[1,1].set_ylim(adjust_margins((0, 1), ax[1,1], percent=0.015))

ax[0,1].set_box_aspect(1)
ax[1,1].set_box_aspect(1)

ax[0,0].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[0,1].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[1,0].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[1,1].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)

ax[1,0].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
ax[1,1].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))

# Create a legend for files (colors only) at a single point in the figure
handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label = simulation_inputs["Diffusion coefficient"]
[i]) for i in range(len(result_files))]
fig.legend(handles=handles,
          #ncols=2,
          loc=(0.87,0.55),
          title="D / m$^2$s$^{\u2212 1}$",
          fontsize=fs)

plt.savefig(FOLDER+"/MC_evolution.png", dpi=300)
plt.show()

#%% METAL-SYSTEM PLOTTING - SHEAR HISTOGRAM
fs = 9
cm = 1/2.54

FOLDER = "Example_metal_system"

files= glob(FOLDER+"/*/final_stats.csv")

colors = sns.color_palette("viridis", len(files))

fig, axes = plt.subplots(1, len(files), figsize=( 17.2 *cm, 12 * cm), sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95)

for i, (file, ax) in enumerate(zip(files, axes)):
    data = pd.read_csv(file)
    shear_hist = []
    for ii in range(len(data)):
        shear_hist.append(np.fromstring(data["Shear Hist Horizontal"].values[ii].strip('[]'), sep=' ', dtype=int))
    shear_hist = np.array(shear_hist)
    lines = np.arange(len(shear_hist[0]))
        
    # Plot the lower bound (value - std) with base color
    ax.plot(shear_hist.mean(axis=0), lines, color=colors[i], alpha=1.0)
    
    ax.fill_betweenx(lines,
                    shear_hist.mean(axis=0) - shear_hist.std(axis=0),
                    shear_hist.mean(axis=0) + shear_hist.std(axis=0),
                    color=colors[i],
                    alpha=0.5)
    
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(labelsize=fs-2)
    #ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    
    ax.set_ylim(len(shear_hist[0]), 0)
    D = simulation_inputs["Diffusion coefficient"]
    ax.set_title(f"$D$ = {D[i]:.1e}", size=fs, rotation=45)
    
    if i > 0 and i < len(files)-1:
        # Hide the right and top spines
        for spine in ['right', 'left']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(left=False, top=True)
            
    elif i == 0:
        # Hide the right and top spines
        for spine in ['right']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(top=True)
            
    else:
        # Hide the right and top spines
        for spine in ['left']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(top=True, right=True, left=False)
            
            
axes[0].set_ylabel("$Grid$ $Line$ / unitless", size=fs)
axes[round(len(files)/2)].set_xlabel("$Shear$ $Count$ / unitless", size=fs)

#plt.tight_layout()
plt.show()
plt.savefig(FOLDER+"/Shear Step Histogramm.png", dpi=300)

#%% METAL-SYSTEM PLOTTING - GENERATE VIDEO

FOLDER = "Example_metal_system"

# Create a custom colormap and swap colors
colors = sns.color_palette("deep", 4)
colors = colors[0], colors[3]  # Swap colors
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

for Dir, Diff in zip(glob(FOLDER+"/*_runs"), simulation_inputs["Diffusion coefficient"]):

    # Define paths
    image_list = glob(Dir + "/for_video/run_0/*.npy")
    
    # Skip video generation if data is not available
    if not image_list:
        continue 
    
    simulation_results =  Dir + "/simulation_results.csv"
    simulation_results = pd.read_csv(simulation_results)
    simulation_results[simulation_results["Run"] == 1]
    output_folder = Dir
    
    # Create a video from the images with swapped colors and exact matplotlib colormap
    create_video_from_arrays(image_list, output_folder + f'/output_video_run_{os.path.splitext(image_list[0])[0][-8]}.mp4', fps=10, cmap=cmap, scale_factor=10, stats_df=simulation_results)
    


#%% METAL-HYDROGEN SYSTEM - DEFINITIONS (REQUIRED for Simulation and Plotting)

# Definitions for plotting - fontsizes & colors

# Create a custom color map for the metal grid using blue and red tones
colors = sns.color_palette("deep", 4)
colors = colors[0], colors[3]
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Define the color map for hydrogen grid using violet and white
colors_H2 = [(1, 1, 1), (112/255, 48/255, 160/255)]
cmap_H2 = ListedColormap(colors_H2)

# Font size
fs = 10

################################################################################################
# Definitions of grid size, number of simulationm and the convergence criteria
# Grid size
grid_size = 50
starting_conf = "Real_starting_conf.png"

# Define the (maximum) number of Monte Carlo steps per simulation and number of repetitions
n_mc_steps = 100000000
n_simulations = 3

# Define convergence criteria for the MC simulation
# Parameters for stopping criterion
n_persistence = 30          # Number of consecutive stable windows
threshold_mean = 0.005      # Threshold for change in mean of mixing index

################################################################################################


################################################################################################
# Definiations of shear probabilities, interaction energies and the "temperature" of the system (300K)

# Shear probabilities for every MC step - the higher the diffusion coefficient the lower the probability for shear
# 50x50 Grid - HPT deformation speed of 1.27 rpm - For different diffusion coefficients
p_shear_list = [0.00000045, # 10-15 m²/s
                0.00000447, # 10-16 m²/s
                0.00004474, # 10-17 m²/s
                0.00044719, # 10-18 m²/s
                0.00445399, # 10-19 m²/s
                0.04282333, # 10-20 m²/s
                0.30910222, # 10-21 m²/s
                0.81731560, # 10-22 m²/s
                0.97813692, # 10-23 m²/s
                0.99776981, # 10-24 m²/s
                0.99977653, # 10-25 m²/s
                0.99997765] # 10-26 m²/s

total_diffusion_probability = 1 - np.array(p_shear_list)

p_normalization_factor = ((1/11) * np.array(p_shear_list)) +  ((1/11) * total_diffusion_probability) +  ((10/11) * total_diffusion_probability)

simulation_inputs = {
                    "Diffusion coefficient"            :  [1e-15, 1e-16, 1e-17, 1e-18, 1e-19, 1e-20, 1e-21, 1e-22, 1e-23, 1e-24, 1e-25, 1e-26],
                    "Shear probability"                : (1/11) * np.array(p_shear_list) / p_normalization_factor,
                    "Metal diffusion probability"      : (1/11) * total_diffusion_probability / p_normalization_factor,
                    "Hydrogen diffusion probability"   : (10/11) * total_diffusion_probability / p_normalization_factor
                    }

p_normalization_factor = simulation_inputs["Shear probability"] +  simulation_inputs["Metal diffusion probability"] + simulation_inputs["Hydrogen diffusion probability"]

simulation_inputs = pd.DataFrame(simulation_inputs)

# Energy parameters and assocaited (assumed) standard deviations
# Interaction energies between A-A, B-B and A-B atoms as well as  A-H and B-H atoms
# Factor 1000 - kJ to J
# Factor 1/4 - Complete mixing ethalpy only obtained when bonding with e.g. 4 different atoms, i.e., 4 A-B pairs,
# Factor 2 due to the douple counting issue 
E_AA_mean = 2 * (1000/4) *  2.66E-25    # Enthalpy of formation TiVZrNbHf                               - kJ atom-1      
E_BB = 0                 # Enthalpy of formation Cu - Defined as zero                    - kJ atom-1 
E_AB_mean = 2 * (1000/4) * -1.21E-23    # Enthalpy of formation TiVZrNbHf-Cu - Real atomic composition  - kJ atom-1     

# Factor 1000 - kJ to J
# Factor 1/4 - Complete mixing ethalpy only obtained when bonding with e.g. 4 different atoms, i.e., 4 A-B pairs,
E_AH_mean      = (1000/4) * -4.90E-23
E_BH_mean      = (1000/4) *  7.05E-23

Estd = False
E_AA_std       = abs(E_AA_mean) * Estd
E_AB_std       = abs(E_AB_mean) * Estd
E_AH_std       = abs(E_AH_mean) * Estd
E_BH_std       = abs(E_BH_mean) * Estd

# Define kT for HPT conditions close to room temperature
kT = constants.Boltzmann * 300

################################################################################################

#%% METAL-HYDROGEN SYSTEM - MONTE CARLO SIMULATION WITH VARYING DIFFUSION COEFFICIENTS
########################### ACTUAL SIMULATION ################################


# LOOP 1 - Iterate through the shear probabilities associated with the different diffusion coefficients
#          i.e., Results as a FUNCTION OF DIFFUSION COEFFICIENT 
for index, simulation_input in simulation_inputs.iterrows():
    
    p_shear              = simulation_input["Shear probability"]
    D                    = simulation_input["Diffusion coefficient"]
    p_metal_diffusion    = simulation_input["Metal diffusion probability"]
    p_hydrogen_diffusion = simulation_input["Hydrogen diffusion probability"]
    
    # Create a folder for the results
    folder_name = f"D_{D}_MC_{n_mc_steps}_steps_{n_simulations}_runs"
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
    
    ##############################################################################################################
    # Initializing all the DataFrames, Lists and other containers for saving all the relevant information of the
    # MC simulation for evaluation and plotting.
    
    # Track the simulation results for every run and step, the latter is sample according to "get_sampling_interval"
    # The total energy and mixing index is saved as a function of steps allowing the plot the evolution of the grid during 
    # simulation
    all_simulation_data = []
    metal_grids = []
    hydrogen_grids = []
    
    # Initial all the required empty list for saving all the data generated during the MC simulation
    diffusion_success_rates = []
    H_diffusion_success_rates = []
    shear_success_rates = []
    final_stats = []
    
    # List to track when convergence has been achieved
    convergence_step = []
    
    diffs = []
    stds = []
    ##############################################################################################################


    # LOOP 2 - Iterate through the MC runs, each being a unique complete MC simulation with its own samples 
    #          interaction energies; however, all the runs have the same shear and diffusion probabilies determined
    #          by LOOP 1
    print( "#############################################################################")
    print(f"Diffusion coefficient                 : {D:.1e}")
    for run in tqdm(range(n_simulations), desc="Simulations Progress"):
        
        #########################################################################################################
        # Sample the interaction energies and save them for later evaluation
        
        # Sample the energies associated A-A and A-B neighbors from a normal distribution 
        if Estd: 
            E_AA = sample_energy(E_AA_mean, E_AA_std)
            E_AB = sample_energy(E_AB_mean, E_AB_mean)
            E_AH = sample_energy(E_AH_mean, E_AH_std)
            E_BH = sample_energy(E_BH_mean, E_BH_std)
        else:
            E_AA = E_AA_mean
            E_AB = E_AB_mean
            E_AH = E_AH_mean
            E_BH = E_BH_mean
        #########################################################################################################

        #########################################################################################################
        # Initializing all the DataFrames, Lists and other containers for saving all the relevant information of the
        # MC simulation for evaluation and plotting.
        
        # Initialize the MC step counters for shear, metal diffusion, and hydrogen diffusion
        successful_metal_diffusion_attempts = 0
        total_metal_diffusion_attempts = 0
        
        successful_hydrogen_diffusion_attempts = 0
        total_hydrogen_diffusion_attempts = 0
        
        successful_shear_attempts = 0
        
        # Initialize the counting for shearing - 1D Numpy Array - used for checking for any kind of shear localization 
        shear_counts_horizontal = np.zeros(grid_size, dtype=np.int64)
        shear_counts_vertical   = np.zeros(grid_size, dtype=np.int64)
        shear_type              = 2 # 0 = Horizontal / 1 = Vertical / 2 = Both
        strength_A              = 1
        strength_B              = 0.2
        max_attempts            = 100
        
        # Initialize the grids - Both metal and hydrogen subgrids
        metal_grid = initialize_grid_from_image(starting_conf, grid_size)
        hydrogen_grid = 1 - metal_grid
        metal_grid = metal_grid.astype(np.int64)
        hydrogen_grid = hydrogen_grid.astype(np.int64)
        hydrogen_positions = np.argwhere(hydrogen_grid == 1) 
        
        # Lists to track energy and mixing index
        energy_samples = []
        mixing_index_samples = []
        
        # Initiate the convergence criteria for stopping the MC simulation
        converged = False  # Flag to indicate convergence - Reset for before every MC run (i.e., for LOOP 3)
        convergence_counter = 0 # Counter for steps fullfilling the convergence - Reset for before every MC run (i.e., for LOOP 3)
        #########################################################################################################

        # Use NumPy’s random function for efficiency
        rng = np.random.default_rng()
        
        sampling_interval = 5
        
        total_energy_list = []
        mixing_index_list = []
        
        step_time = []
        interval_time = []
        diff_time = []
        shear_time = []
        
        
        # Iterate through the MC steps 
        for step in range(n_mc_steps): #range(n_mc_steps)
            
            random_value = rng.random(1)
            
            #start_time = time.time()
            #start_time_shear = time.time()
            
            # Metal diffusion step
            if random_value < p_metal_diffusion:
                # Count all diffusion attempts
                total_metal_diffusion_attempts += 1
                if metal_diffusion_step_with_hydrogen(metal_grid,
                                                      hydrogen_grid,
                                                      kT=kT,
                                                      E_AA=E_AA,
                                                      E_BB=E_BB,
                                                      E_AB=E_AB,
                                                      E_AH=E_AH,
                                                      E_BH=E_BH,
                                                      grid_size=grid_size):
                    successful_metal_diffusion_attempts += 1
                    
            #shear_time.append(time.time() - start_time_shear)
            #start_time_diff = time.time()
            
            # Hydrogen diffusion step
            if random_value < p_hydrogen_diffusion:
                # Count all diffusion attempts
                total_hydrogen_diffusion_attempts += 1
                if hydrogen_diffusion_step(hydrogen_grid,
                                           metal_grid,
                                           kT=kT,
                                           E_AH=E_AH,
                                           E_BH=E_BH,
                                           grid_size=grid_size,
                                           hydrogen_positions=hydrogen_positions):
                    successful_hydrogen_diffusion_attempts += 1

            # Shear step
            if random_value < p_shear:
                # Count all shear attempts
                successful_shear_attempts += 1
                # Shear attempts are always successfull
                shear_step_with_hydrogen(metal_grid,
                                         hydrogen_grid,
                                         shear_counts_horizontal,
                                         shear_counts_vertical,
                                         shear_type,
                                         strength_A,
                                         strength_B,
                                         max_attempts)
            
            #diff_time.append(time.time() - start_time_diff)
            #step_time.append(time.time() - start_time)
            
            
            # If the MC step is a multiple of the sampling interval then the energy and mixing index is calulated and saved
            if step % sampling_interval == 0:
                #start_time = time.time()
                # Sample the systems energy and ordering in increasingly coarser steps 
                # i.e. lots of sampling initially and spare sampling at larger MC step counts
                sampling_interval = get_sampling_interval(step)
                
                total_energy = calculate_total_metal_metal_interaction_energy(metal_grid, E_AA, E_BB, E_AB) + calculate_total_metal_hydrogen_interaction_energy(metal_grid, hydrogen_grid, E_AH, E_BH)
                mixing_index = calculate_mixing_index(metal_grid)
                total_energy_list.append(total_energy)
                mixing_index_list.append(mixing_index)
                
                
                video_folder = f"{folder_name}/for_video/run_{run}"
                if not os.path.exists(video_folder):
                    os.makedirs(video_folder)
                np.save(f"{folder_name}/for_video/run_{run}/metal_grid_run_{run}_step_{step}", metal_grid)
                np.save(f"{folder_name}/for_video/run_{run}/hydrogen_grid_run_{run}_step_{step}", hydrogen_grid)
                
                
                all_simulation_data.append([run, step, total_energy, mixing_index,
                                            successful_shear_attempts, total_metal_diffusion_attempts, successful_metal_diffusion_attempts,
                                            total_hydrogen_diffusion_attempts, successful_hydrogen_diffusion_attempts])
                
                #################################################################################################
                # CHECK CONVERGENCE CRITERIA
                
                # Implementation of convergence criteria
                # Check for convergence every `n_window` steps
                if not converged and step > 100: #and step % sampling_interval == 0 :
                    # Calc the mean of the last 10 simulation results (to reduce effect of random variation)
                    _current = np.mean(mixing_index_list[-5:])
                    std_current = np.mean(mixing_index_list[-5:])
                    
                    # Calc the mean of the simulation results a little further away
                    _previous = np.mean(mixing_index_list[-20:-15])
                    
                    # Calc the changes in the mixing index for checking convergence
                    diff = abs(_current - _previous)
                    
                    # Save the values calculated for the convergence criteria for plotting / evaluating the convergence 
                    # all_simulation_data["Mixing Index Difference"].iat[-1] = diff
                    # all_simulation_data["Mixing Index Std"].iat[-1] = std_current
                    all_simulation_data[-1].append(diff)
                    all_simulation_data[-1].append(std_current)
                    
                    if step > 1000000:
                        # CHECK FOR CONVERGENCE - IF CONVERGED -> ADD TO THE CONVERGENCE COUNTER
                        if diff < threshold_mean:
                            convergence_counter += 1
          
                        # IF NOT CONVERGED - RESET THE CONVERGENCE COUNTER TO ZERO
                        else:
                            convergence_counter = 0  # Reset if stability is broken
                    
                        # Stop the simulation if stable for `n_persistence` windows
                        if convergence_counter >= n_persistence:
                            print(f"Convergence reached at step {step}")
                            converged = True
                            continuation_counter = np.ceil(step/10)
                            convergence_step.append(step)
                
                #interval_time.append(time.time() - start_time)
                ######################################################################################################
                        
            #####################################################################################################
            # ENFORCE CONVERGENCE - after the simulation has convergened and the coninuation counter is reached
        
            # Handle continuation phase - Continue MC simualtion for `continuation_counter` amount of steps
            if converged:
                if continuation_counter > 0:
                    continuation_counter -= 1
                else:
                    print(f"Continuation phase completed. Stopping simulation at step {step}")
                    break
                
            #####################################################################################################
        
        # If no converence is reached use the last MC step as the `convergence step`,
        # i.e. the step at which convergence was reached             
        if not converged:
            convergence_step.append(step)
            
        #########################################################################################################

        # np.save(f"{folder_name}/step_time_run_{run}", np.array(step_time))
        # np.save(f"{folder_name}/interval_time_run_{run}", np.array(interval_time))
        # np.save(f"{folder_name}/shear_time_run_{run}", np.array(shear_time))
        # np.save(f"{folder_name}/diff_time_run_{run}", np.array(diff_time))


               
        # After each run, calculate the attempt and success rates of both diffusion and shear
        # Metal diffusion attemps
        fraction_metal_diffusion_steps = total_metal_diffusion_attempts / step
        # Successful metal diffusion attemps - passed the Metropolis criterion
        fraction_successful_metal_diffusion_steps = successful_metal_diffusion_attempts / step
        # Hydrogen diffusion attemps
        fraction_hydrogen_diffusion_steps = total_hydrogen_diffusion_attempts / step
        # Successful hydrogen diffusion attemps - passed the Metropolis criterion
        fraction_successful_hydrogen_diffusion_steps = successful_hydrogen_diffusion_attempts / step
        # Shear attemps = Successful shear attempt - Shear not goverend by interaction energies and thermodynamics
        fraction_shear_steps = successful_shear_attempts / step
    
        # Save the calculated attempt and success rates for each run
        diffusion_success_rates.append(fraction_successful_metal_diffusion_steps)
        H_diffusion_success_rates.append(fraction_successful_hydrogen_diffusion_steps)
        shear_success_rates.append(fraction_shear_steps)

        # Calculate mean and std for the recorded values of total energy and the mixing index in the persistence window 
        # Values should have converged already in this window - Save these final stats for each run
        mean_total_energy = np.mean(total_energy_list[-n_persistence:])
        std_total_energy  = np.std( total_energy_list[-n_persistence:])
        mean_mixing_index = np.mean(mixing_index_list[-n_persistence:])
        std_mixing_index  = np.std( mixing_index_list[-n_persistence:])
        
        # Save all the final stats of the 
        final_stats.append({
            'Run'                                     : run,                                     # Which MC run was this
            'p_metal_diffusion'                       : p_metal_diffusion,                       # Nominal metal diffusion probability
            'p_hydrogen_diffusion'                    : p_metal_diffusion,                       # Nominal hydrogen diffusion probability
            'p_shear'                                 : p_shear,                                 # Nominal shear probability
            'E_AA'                                    : E_AA,                                    # A-A interaction energy during this run
            'E_AA'                                    : E_AA,                                    # B-B interaction energy during this run
            'E_AA'                                    : E_AA,                                    # A-B interaction energy during this run
            'E_AH'                                    : E_AH,                                    # A-H interaction energy during this run
            'E_BH'                                    : E_BH,                                    # B-H interaction energy during this run
            'Mean Total Energy'                       : mean_total_energy,                       # Mean energy at the end of the simulation
            'Std Total Energy'                        : std_total_energy,                        # Std of the energy at the end of the simulation
            'Mean Mixing Index'                       : mean_mixing_index,                       # Mean mixing index at the end of the simulation
            'Std Mixing Index'                        : std_mixing_index,                        # Std of the mixing index at the end of the simulation
            "Steps"                                   : step,                                    # Number performed MC steps
            "Metal Diffusion Attemps"                 : total_metal_diffusion_attempts,          # Diffusion attemps
            "Successful Metal Diffusion Attempts"     : successful_metal_diffusion_attempts,     # Successful diffusion attemps
            "Hydrogen Diffusion Attemps"              : total_hydrogen_diffusion_attempts,       # Diffusion attemps
            "Successful Hydrogen Diffusion Attempts"  : successful_hydrogen_diffusion_attempts,  # Successful diffusion attemps
            "Successful Shear Attempts"               : successful_shear_attempts,               # Successful shear attemps
            "Shear Hist Horizontal"                   : shear_counts_horizontal,                 # Save Numpy array saving where shear steps were made
            "Shear Hist Vertical"                     : shear_counts_horizontal                  # Save Numpy array saving where shear steps were made 
            })
        
        
        np.save(f"{folder_name}/metal_grid_run_{run}", metal_grid)
        np.save(f"{folder_name}/hydrogen_grid_run_{run}", hydrogen_grid)
        metal_grids.append(metal_grid)
        hydrogen_grids.append(hydrogen_grid)
        
        #########################################################################################################
        
    # Calculate mean and standard deviation for success rates
    mean_metal_diffusion_success_rate    = np.mean(fraction_successful_metal_diffusion_steps)
    std_metal_diffusion_success_rate     = np.std( fraction_successful_metal_diffusion_steps)
    mean_hydrogen_diffusion_success_rate = np.mean(fraction_successful_hydrogen_diffusion_steps)
    std_hydrogen_diffusion_success_rate  = np.std( fraction_successful_hydrogen_diffusion_steps)
    mean_shear_rate                      = np.mean(shear_success_rates)
    std_shear_rate                       = np.std( shear_success_rates)

    # Print the mean and std of success rates
    
    print(f"Mean metal diffusion attempt rate     : {fraction_metal_diffusion_steps:.4f}")
    print(f"Mean hydrogen diffusion attempt rate  : {fraction_hydrogen_diffusion_steps:.4f}")
    print(f"Mean shear attempt rate               : {fraction_shear_steps:.4f}")


    print(f"Mean metal diffusion success rate     : {mean_metal_diffusion_success_rate:.4f} ± {std_metal_diffusion_success_rate:.4f}")
    print(f"Mean hydrogen diffusion success rate  : {mean_hydrogen_diffusion_success_rate:.4f} ± {std_hydrogen_diffusion_success_rate:.4f}")
    print(f"Mean shear success rate               : {mean_shear_rate:.4f}             ± {std_shear_rate:.4f}")

    print( "#############################################################################")
    # Save all the simulation data collected during the simulation and the final simulation stats
    final_stats_df = pd.DataFrame(final_stats)
    final_stats_df.to_csv(f'{folder_name}/final_stats.csv', index=False)
    
    all_simulation_data = pd.DataFrame(all_simulation_data, columns=['Run', 'Step', 'Total Energy', 'Mixing Index', 
                                                           'Successful Shear Attempts',
                                                           'Total Metal Diffusion Attempts', 'Successful Metal Diffusion Attempts',
                                                           'Total Hydrogen Diffusion Attempts', 'Successful Hydrogen Diffusion Attempts',
                                                           'Mixing Index Difference','Mixing Index Std'])
    all_simulation_data.to_csv(f'{folder_name}/simulation_results.csv', index=False)

    
    ##############################################################################################################
    # Plotting all results in a single masterplot for every MC run - All parameters are plotted as a fucntion of MC steps
    for i, c in enumerate(np.array(convergence_step)):
        simulation_data = all_simulation_data.loc[all_simulation_data["Run"] == i]
        metal_grid = metal_grids[i]
        hydrogen_grid = hydrogen_grids[i]
        
        fig, ax = plt.subplots(3,2, figsize=(12, 12), facecolor='white')
        plt.subplots_adjust(left=0.05, bottom=0.1, right=0.70, top=0.9, wspace=0.1)
        
        colors = sns.color_palette("deep", 4)    

        ax[0,0].imshow(metal_grid, cmap=cmap, origin='upper')
        ax[0,0].set_xlabel("$x$ / unitless")
        ax[0,0].set_ylabel("$y$ / unitless")
        
        ax[1,0].imshow(hydrogen_grid, cmap=cmap_H2, origin='upper')
        ax[1,0].set_xlabel("$x$ / unitless")
        ax[1,0].set_ylabel("$y$ / unitless")

        ax[2,0].semilogx(simulation_data["Step"], simulation_data["Total Energy"], marker="o", linestyle="", color = colors[0], label="Total Energy")
        ax[2,0].semilogx(simulation_data["Step"].iloc[-1], simulation_data["Total Energy"].iloc[-1],
                        marker="o", markersize=6, color="black")
        ax[2,0].set_ylabel("$Total$ $Energy$ / arb.u.")
        ax[2,0].set_box_aspect(1)

        ax_twin = ax[2,0].twinx()
        ax_twin.semilogx(simulation_data["Step"], simulation_data["Mixing Index"], marker="o", linestyle="", color = colors[1], label="Mixing Index")
        ax_twin.semilogx(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index"].iloc[-1],
                         marker="o", markersize=6, color="black")
        ax_twin.set_xlabel("$Monte$ $Carlos$ $Step$ / unitless")
        ax_twin.set_ylabel("$Mixing$ $Index$ / unitless")
        ax_twin.set_box_aspect(1)
        ax_twin.legend()

        ax[0,1].semilogx(simulation_data["Step"], simulation_data["Mixing Index Difference"], color = colors[0], label="Mean")
        ax[0,1].semilogx(simulation_data["Step"], simulation_data["Mixing Index Std"], marker="o", linestyle="", color = colors[1], label="Std")
        ax[0,1].semilogx(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index Difference"].iloc[-1],
                          marker="o", markersize=6, color="black")
        ax[0,1].axhline(threshold_mean, linestyle="--", color="black")
        ax[0,1].set_ylabel("$Mixing$ $Index$ / unitless")
        ax[0,1].set_box_aspect(1)
        ax[0,1].legend(loc="upper left", bbox_to_anchor=[1.05, 1])

        ax[1,1].loglog(simulation_data["Step"], simulation_data["Successful Shear Attempts"], marker="^", linestyle="", color = colors[0], label="Successful shear attempts")
        ax[1,1].loglog(simulation_data["Step"], simulation_data["Total Metal Diffusion Attempts"], marker="v", linestyle="", color = colors[1], label="Metal diffusion attempts")
        ax[1,1].loglog(simulation_data["Step"], simulation_data["Successful Metal Diffusion Attempts"], marker="^", linestyle="", color = colors[1], label="Successful metal diffusion attempts")
        ax[1,1].loglog(simulation_data["Step"], simulation_data["Total Hydrogen Diffusion Attempts"], marker="v", linestyle="", color = colors[2], label="Hydrogen diffusion attempts")
        ax[1,1].loglog(simulation_data["Step"], simulation_data["Successful Hydrogen Diffusion Attempts"], marker="^", linestyle="", color = colors[2], label="Successful hydrogen diffusion attempts")
        ax[1,1].set_ylabel("$Attemps$ / unitless")
        ax[1,1].set_box_aspect(1)
        ax[1,1].legend(loc="upper left", bbox_to_anchor=[1.05, 1])

        ax[2,1].loglog(simulation_data["Step"], simulation_data["Successful Shear Attempts"]                 / (simulation_data["Step"]+1), marker="^", linestyle="", color = colors[0], label="Successful shear attempts") 
        ax[2,1].loglog(simulation_data["Step"], simulation_data["Total Metal Diffusion Attempts"]    / (simulation_data["Step"]+1), marker="v", linestyle="", color = colors[1], label="Metal diffusion attempts")
        ax[2,1].loglog(simulation_data["Step"], simulation_data["Successful Metal Diffusion Attempts"]  / (simulation_data["Step"]+1), marker="^", linestyle="", color = colors[1], label="Successful metal diffusion attempts")    
        ax[2,1].loglog(simulation_data["Step"], simulation_data["Total Hydrogen Diffusion Attempts"]    / (simulation_data["Step"]+1), marker="v", linestyle="", color = colors[2], label="Hydrogen diffusion attempts")
        ax[2,1].loglog(simulation_data["Step"], simulation_data["Successful Hydrogen Diffusion Attempts"]  / (simulation_data["Step"]+1), marker="^", linestyle="", color = colors[2], label="Successful hydrogen diffusion attempts")      
        ax[2,1].set_xlabel("$Monte$ $Carlos$ $Step$ / unitless")
        ax[2,1].set_ylabel("$Attempt$ $ratios$ / unitless")
        ax[2,1].set_box_aspect(1)
        ax[2,1].legend(loc="upper left", bbox_to_anchor=[1.05, 1])

        fig.suptitle(f'Monte Carlo Run {i} - D={D} m$^{2}$/s', fontsize=12)

        plt.savefig(f'{folder_name}/simulation_results_run_{i}.png', dpi=300)
    ##############################################################################################################
    

    ##############################################################################################################
    # LINEAR PLOT - Plot results all MC runs for this diffusion coefficient
    fig, ax = plt.subplots(2, 1, figsize=(8.5*cm,  15*cm))
    fig.subplots_adjust(left=0.2, right=0.75)
    
    colors = sns.color_palette("viridis", n_simulations)
    
    # Plot the energy and mixing index for every MC run as a function of MC steps - mark the convergence point with a "o"
    for i, c in enumerate(np.array(convergence_step)):
        simulation_data = all_simulation_data.loc[all_simulation_data["Run"] == i]
        color = colors[i]
        ax[0].plot(simulation_data["Step"], simulation_data["Total Energy"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].plot(simulation_data["Step"], simulation_data["Mixing Index"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].plot(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index"].iloc[-1],
                    label=f'Run {i}', marker="o", markersize=6, color=color)
        
    # Axis labels and color bar
    ax[0].set_ylabel('$Total$ $Energy$ / arb. u.', size=fs)
    ax[1].set_xlabel('$Monte$ $Carlo$ $step$ / unitless', size=fs)
    ax[1].set_ylabel('$Mixing$ $Index$ / unitless', size=fs)
    
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    
    # Add a colorbar related to the MC runs
    cax = fig.add_axes([ax[1].get_position().x1+0.05,ax[1].get_position().y0,0.05,ax[0].get_position().y1-ax[1].get_position().y0])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_simulations - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", pad=0.05)
    cbar.set_label('Run Number', size=fs)
    
    # Save the plot
    plt.savefig(f'{folder_name}/simulation_results_linear.png', dpi=300)
    ##############################################################################################################
    
    
    ##############################################################################################################
    # LOG PLOT - Plot results all MC runs for this diffusion coefficient
    fig, ax = plt.subplots(2, 1, figsize=(8.5*cm,  15*cm))
    fig.subplots_adjust(left=0.2, right=0.75)
    
    colors = sns.color_palette("viridis", n_simulations)
     
    # Plot the energy and mixing index for every MC run as a function of MC steps - mark the convergence point with a "o"
    for i, c in enumerate(np.array(convergence_step)):
        simulation_data = all_simulation_data.loc[all_simulation_data["Run"] == i]
        color = colors[i]
        ax[0].semilogx(simulation_data["Step"], simulation_data["Total Energy"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].semilogx(simulation_data["Step"], simulation_data["Mixing Index"], label=f'Run {i}', color=color, alpha=0.5)
        ax[1].semilogx(simulation_data["Step"].iloc[-1], simulation_data["Mixing Index"].iloc[-1],
                   label=f'Run {i}', marker="o", markersize=6, color=color)
        
    # Axis labels and color bar
    ax[0].set_ylabel('$Total$ $Energy$ / arb. u.', size=fs)
    ax[1].set_xlabel('$Monte$ $Carlo$ $step$ / unitless', size=fs)
    ax[1].set_ylabel('$Mixing$ $Index$ / unitless', size=fs)
    
    ax[0].set_box_aspect(1)
    ax[1].set_box_aspect(1)
    
    # Add a colorbar related to the MC runs
    cax = fig.add_axes([ax[1].get_position().x1+0.05,ax[1].get_position().y0,0.05,ax[0].get_position().y1-ax[1].get_position().y0])
    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(vmin=0, vmax=n_simulations - 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical", pad=0.05)
    cbar.set_label('Run Number', size=fs)
    
    # Save the plot
    plt.savefig(f'{folder_name}/simulation_results_log.png', dpi=300)
    
    plt.close("all")
    ##############################################################################################################

#%% METAL-HYDROGEN-SYSTEM PLOTTING - MIXING INDEX & TOTAL ENERGY vs DIFFUSION COEFFICIENT - SUPPLEMENTARY FIGURE

FOLDER = "Example_metal_hydrogen_system"

# Gather CSV files
files = glob(FOLDER+"/*/final_stats.csv")
# Reorganize the order of files, ensuring correct concatenation


# Set up colormap
cmap = sns.color_palette("viridis", as_cmap=True)

# Create subplots
fig, ax = plt.subplots(2, 1, figsize=(5, 10), sharex=True)

# Iterate through shear probabilities, diffusion coefficients, and file paths
for i, (p_shear, D, file) in enumerate(zip(simulation_inputs["Shear probability"], simulation_inputs["Diffusion coefficient"], files)):
    
    # Read data from the current CSV file
    stats = pd.read_csv(file)
    
    # Map color based on the mean of "Successful Shear Attempt" using a logarithmic scale
    color = cmap(mcolors.LogNorm(vmin=0.1, vmax=8000 * 100)(stats["Successful Shear Attempts"].mean()))
    
    # Plot the mean and standard deviation of the Mixing Index
    ax[0].errorbar(
        D,
        stats["Mean Mixing Index"].mean(),
        yerr=stats["Mean Mixing Index"].std(),
        marker="o",
        linestyle="-",
        capsize=3,
        capthick=2,
        color=color
    )

# Second loop to plot the Total Energy
for i, (p_shear, D, file) in enumerate(zip(simulation_inputs["Shear probability"], simulation_inputs["Diffusion coefficient"], files)):
    
    # Read data from the current CSV file
    stats = pd.read_csv(file)
    
    # Map color based on the mean of "Successful Shear Attempt" using a logarithmic scale
    color = cmap(mcolors.LogNorm(vmin=0.1, vmax=8000 * 100)(stats["Successful Shear Attempts"].mean()))
    
    # Plot the mean and standard deviation of the Total Energy, scaled by Avogadro's constant
    ax[1].errorbar(
        D,
        (constants.Avogadro * stats["Mean Total Energy"].mean()) / (1000 * 50**2),  # Convert to kJ/mol for a 50x50 grid
        yerr=(constants.Avogadro * stats["Mean Total Energy"].std()) / (1000 * 50**2),
        marker="o",
        linestyle="-",
        capsize=3,
        capthick=2,
        color=color
    )

# Configure the first subplot (Mixing Index)
ax[0].set_xscale('log')  # Set x-axis to logarithmic scale
ax[1].set_xlabel("$D$ / m$^{2}$ s$^{-1}$")  # Label for x-axis
ax[0].set_ylabel("$Mean$ $Mixing$ $Index$ / unitless")  # Label for y-axis
ax[0].set_xlim((10**-30, 10**-12))  # Set x-axis limits
ax[0].set_ylim((0, 1))  # Set y-axis limits

# Configure the second subplot (Total Energy)
ax[1].set_ylabel("\u0394$H_{mix}$ / kJ mol$^{\u22121}$")  # Label for y-axis
ax[1].set_ylim((-20, 0))  # Set y-axis limits

# Create a logarithmic colorbar for the number of Successful Shear Attempts
sm = plt.cm.ScalarMappable(cmap="viridis", norm=mcolors.LogNorm(vmin=0.1, vmax=8000 * 100))
sm.set_array([])
cbar = fig.colorbar(sm, ax=ax, orientation="vertical", pad=0.05)
cbar.set_label('$Successful$ $Shear$ $Attempt$ / unitless', size=12)

# Save the figure
plt.savefig(FOLDER+"/MixingIndex_Energy_vs_DiffusionCoeff.png", dpi=300)


#%% METAL-HYDROGEN-SYSTEM PLOTTING - LIN - MIXING INDEX & TOTAL ENERGY vs MC STEPS & STRAIN - SUPPLEMENTARY FIGURE

FOLDER = "Example_metal_hydrogen_system"

# Define font size and figure size
fs = 9
fig, ax = plt.subplots(2, 2, figsize=(17.2*cm, 15.0*cm), sharex='col')
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95)

# Define the file path pattern for all files you want to include
file_pattern = FOLDER+"/*/simulation_results.csv"

# Get list of result and stats files
result_files = sorted(glob(file_pattern))

# Set up color map
colors = sns.color_palette("viridis", len(result_files))

# Loop through each file
for j, (result_file, D) in enumerate(zip(result_files, simulation_inputs["Diffusion coefficient"])):
    
    # Load the data
    results_data = pd.read_csv(result_file)
    
    # Group by 'Step' to calculate mean and std for 'MixingIndex' and 'Energy'
    result_mean_std = results_data.groupby('Step').agg({
        'Mixing Index': ['mean', 'std'],
        'Total Energy': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns for cleaner column names
    result_mean_std.columns = ['Step', 'MixingIndex_mean', 'MixingIndex_std', 'TotalEnergy_mean', 'TotalEnergy_std']

    result_mean_std["TotalEnergy_mean_molar"] = (constants.Avogadro * result_mean_std["TotalEnergy_mean"]) / (1000 * 50**2)
    result_mean_std["TotalEnergy_std_molar"] = (constants.Avogadro * result_mean_std["TotalEnergy_std"]) / (1000 * 50**2)
    
    # Plot with color based on file index
    color = colors[j]
    ax[0,0].plot(result_mean_std["Step"],
                   result_mean_std["TotalEnergy_mean_molar"],
                   color=color,
                   alpha=1,
                   marker="",
                   linestyle="-",
                   label=str(D))
    
    ax[0,0].fill_between(result_mean_std["Step"],
                       result_mean_std["TotalEnergy_mean_molar"] - result_mean_std["TotalEnergy_std_molar"],
                       result_mean_std["TotalEnergy_mean_molar"] + result_mean_std["TotalEnergy_std_molar"],
                       color=color,
                       alpha=0.2)
    
    #ax[0,0].set_xscale('log')
    
    ax[1,0].plot(result_mean_std["Step"],
                   result_mean_std["MixingIndex_mean"],
                   color=color,
                   alpha=1,
                   marker="",
                   linestyle="-",
                   label=str(D))
    
    ax[1,0].fill_between(result_mean_std["Step"],
                       result_mean_std["MixingIndex_mean"] - result_mean_std["MixingIndex_std"],
                       result_mean_std["MixingIndex_mean"] + result_mean_std["MixingIndex_std"],
                       color=color,
                       alpha=0.2)
    #ax[1,0].set_xscale('log')
    
    
    results_data['Successful Shear Attempts Grouped'] = results_data['Successful Shear Attempts'].apply(dynamic_grouping)   
    
    # Group by 'Step' to calculate mean and std for 'MixingIndex' and 'Energy'
    result_straingrouped = results_data.groupby('Successful Shear Attempts Grouped').agg({
        'Mixing Index': ['mean', 'std'],
        'Total Energy': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns for cleaner column names
    result_straingrouped.columns = ['Successful Shear Attempts Grouped', 'MixingIndex_mean', 'MixingIndex_std', 'TotalEnergy_mean', 'TotalEnergy_std']

    # Convert the Total Energy to the Total Molar Energy i.e. kJ mol-1
    result_straingrouped["TotalEnergy_mean_molar"] = (constants.Avogadro * result_straingrouped["TotalEnergy_mean"]) / (1000 * 50**2)
    result_straingrouped["TotalEnergy_std_molar"]  = (constants.Avogadro * result_straingrouped["TotalEnergy_std"]) / (1000 * 50**2)
    
    # Plot with color based on file index
    # Divide number of Successful Shear Attempts by 50 to get the actuall shear strain - 50 Shear Steps are a Shear Strain of 1.
    color = colors[j]
    ax[0,1].plot(result_straingrouped["Successful Shear Attempts Grouped"]/50,
               result_straingrouped["TotalEnergy_mean_molar"],
               color=color,
               alpha=1,
               marker="",
               linestyle="-",
               label=str(D))
    
    ax[0,1].fill_between(result_straingrouped["Successful Shear Attempts Grouped"]/50,
                       result_straingrouped["TotalEnergy_mean_molar"] - result_straingrouped["TotalEnergy_std_molar"],
                       result_straingrouped["TotalEnergy_mean_molar"] + result_straingrouped["TotalEnergy_std_molar"],
                       color=color,
                       alpha=0.2)
    
    #ax[0,1].set_xscale('log')
    
    ax[1,1].plot(result_straingrouped["Successful Shear Attempts Grouped"]/50,
               result_straingrouped["MixingIndex_mean"],
               color=color,
               alpha=1,
               marker="",
               linestyle="-",
               label=str(D))
    
    ax[1,1].fill_between(result_straingrouped["Successful Shear Attempts Grouped"]/50,
                       result_straingrouped["MixingIndex_mean"] - result_straingrouped["MixingIndex_std"],
                       result_straingrouped["MixingIndex_mean"] + result_straingrouped["MixingIndex_std"],
                       color=color,
                       alpha=0.2)
    #ax[1,1].set_xscale('log')

# Set titles, labels, and legends
#ax[0,0].set_xlabel('Strain / unitless', size=fs)
ax[0,0].set_ylabel('\u0394$H_{mix}$ / kJ mol$^{\u22121}$', size=fs)
ax[1,0].set_xlabel('$Monte$ $Carlo$ $Step$ / unitless', size=fs)
ax[1,0].set_ylabel('$Mixing$ $Index$ / unitless', size=fs)


ax[0,0].set_box_aspect(1)
ax[1,0].set_box_aspect(1)

ax[1,0].set_ylim((0,1))

# Set titles, labels, and legends
ax[1,1].set_xlabel('$Shear$ $Strain$ $\u03B3$ / unitless', size=fs)

ax[1,0].set_xlim(adjust_margins((10**1, 8*10**6), ax[1,0], scaling="linear", percent=0.015))
ax[1,1].set_xlim(adjust_margins((10**1, 8*10**4), ax[1,1], scaling="linear", percent=0.015))

ax[0,0].set_ylim(adjust_margins((-15, 0), ax[0,0], percent=0.015))
ax[0,1].set_ylim(adjust_margins((-15, 0), ax[0,1], percent=0.015))
ax[1,0].set_ylim(adjust_margins((0.1, 0.6), ax[1,0], percent=0.015))
ax[1,1].set_ylim(adjust_margins((0.1, 0.6), ax[1,1], percent=0.015))

ax[0,1].set_box_aspect(1)
ax[1,1].set_box_aspect(1)

ax[0,0].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[0,1].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[1,0].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[1,1].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)

# Create a legend for files (colors only) at a single point in the figure
handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label = simulation_inputs["Diffusion coefficient"]
[i]) for i in range(len(result_files))]
fig.legend(handles=handles,
          #ncols=2,
          loc=(0.87,0.55),
          title="D / m$^2$s$^{\u2212 1}$",
          fontsize=fs)

plt.savefig(FOLDER+"/MC_evolution_Linear.png", dpi=300)
plt.show()

#%% METAL-HYDROGEN-SYSTEM PLOTTING - LOG - MIXING INDEX & TOTAL ENERGY vs MC STEPS & STRAIN - SUPPLEMENTARY FIGURE

FOLDER = "Example_metal_hydrogen_system"

# Define font size and figure size
fs = 9
fig, ax = plt.subplots(2, 2, figsize=(17.2*cm, 15.0*cm), sharex="col")
fig.subplots_adjust(left=0.1, bottom=0.1, right=0.85, top=0.95)

# Define the file path pattern for all files you want to include
file_pattern = FOLDER+"/*/simulation_results.csv"

# Get list of result and stats files
result_files = sorted(glob(file_pattern))

# Set up color map
colors = sns.color_palette("viridis", len(result_files))

# Loop through each file
for j, (result_file, D) in enumerate(zip(result_files, simulation_inputs["Diffusion coefficient"])):
    
    # Load the data
    results_data = pd.read_csv(result_file)
    
    # Group by 'Step' to calculate mean and std for 'MixingIndex' and 'Energy'
    result_mean_std = results_data.groupby('Step').agg({
        'Mixing Index': ['mean', 'std'],
        'Total Energy': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns for cleaner column names
    result_mean_std.columns = ['Step', 'MixingIndex_mean', 'MixingIndex_std', 'TotalEnergy_mean', 'TotalEnergy_std']

    result_mean_std["TotalEnergy_mean_molar"] = (constants.Avogadro * result_mean_std["TotalEnergy_mean"]) / (1000 * 50**2)
    result_mean_std["TotalEnergy_std_molar"] = (constants.Avogadro * result_mean_std["TotalEnergy_std"]) / (1000 * 50**2)
    
    # Plot with color based on file index
    color = colors[j]
    ax[0,0].plot(result_mean_std["Step"],
                   result_mean_std["TotalEnergy_mean_molar"],
                   color=color,
                   alpha=1,
                   marker="",
                   linestyle="-",
                   label=str(D))
    
    ax[0,0].fill_between(result_mean_std["Step"],
                       result_mean_std["TotalEnergy_mean_molar"] - result_mean_std["TotalEnergy_std_molar"],
                       result_mean_std["TotalEnergy_mean_molar"] + result_mean_std["TotalEnergy_std_molar"],
                       color=color,
                       alpha=0.2)
    
    ax[0,0].set_xscale('log')
    
    ax[1,0].plot(result_mean_std["Step"],
                   result_mean_std["MixingIndex_mean"],
                   color=color,
                   alpha=1,
                   marker="",
                   linestyle="-",
                   label=str(D))
    
    ax[1,0].fill_between(result_mean_std["Step"],
                       result_mean_std["MixingIndex_mean"] - result_mean_std["MixingIndex_std"],
                       result_mean_std["MixingIndex_mean"] + result_mean_std["MixingIndex_std"],
                       color=color,
                       alpha=0.2)
    ax[1,0].set_xscale('log')
    
    
    results_data['Successful Shear Attempts Grouped'] = results_data['Successful Shear Attempts'].apply(dynamic_grouping)   
    
    # Group by 'Step' to calculate mean and std for 'MixingIndex' and 'Energy'
    result_straingrouped = results_data.groupby('Successful Shear Attempts Grouped').agg({
        'Mixing Index': ['mean', 'std'],
        'Total Energy': ['mean', 'std']
    }).reset_index()

    # Flatten the MultiIndex columns for cleaner column names
    result_straingrouped.columns = ['Successful Shear Attempts Grouped', 'MixingIndex_mean', 'MixingIndex_std', 'TotalEnergy_mean', 'TotalEnergy_std']

    # Convert the Total Energy to the Total Molar Energy i.e. kJ mol-1
    result_straingrouped["TotalEnergy_mean_molar"] = (constants.Avogadro * result_straingrouped["TotalEnergy_mean"]) / (1000 * 50**2)
    result_straingrouped["TotalEnergy_std_molar"]  = (constants.Avogadro * result_straingrouped["TotalEnergy_std"]) / (1000 * 50**2)
    
    # Plot with color based on file index
    # Divide number of Successful Shear Attempts by 50 to get the actuall shear strain - 50 Shear Steps are a Shear Strain of 1.
    color = colors[j]
    ax[0,1].plot(result_straingrouped["Successful Shear Attempts Grouped"]/50,
               result_straingrouped["TotalEnergy_mean_molar"],
               color=color,
               alpha=1,
               marker="",
               linestyle="-",
               label=str(D))
    
    ax[0,1].fill_between(result_straingrouped["Successful Shear Attempts Grouped"]/50,
                       result_straingrouped["TotalEnergy_mean_molar"] - result_straingrouped["TotalEnergy_std_molar"],
                       result_straingrouped["TotalEnergy_mean_molar"] + result_straingrouped["TotalEnergy_std_molar"],
                       color=color,
                       alpha=0.2)
    
    ax[0,1].set_xscale('log')
    
    ax[1,1].plot(result_straingrouped["Successful Shear Attempts Grouped"]/50,
               result_straingrouped["MixingIndex_mean"],
               color=color,
               alpha=1,
               marker="",
               linestyle="-",
               label=str(D))
    
    ax[1,1].fill_between(result_straingrouped["Successful Shear Attempts Grouped"]/50,
                       result_straingrouped["MixingIndex_mean"] - result_straingrouped["MixingIndex_std"],
                       result_straingrouped["MixingIndex_mean"] + result_straingrouped["MixingIndex_std"],
                       color=color,
                       alpha=0.2)
    ax[1,1].set_xscale('log')

# Set titles, labels, and legends
#ax[0,0].set_xlabel('Strain / unitless', size=fs)
ax[0,0].set_ylabel('\u0394$H_{mix}$ / kJ mol$^{\u22121}$', size=fs)
ax[1,0].set_xlabel('$Monte$ $Carlo$ $Step$ / unitless', size=fs)
ax[1,0].set_ylabel('$Mixing$ $Index$ / unitless', size=fs)

#ax[0,0].set_xlim(adjust_margins((10**1, 10**8), ax[0,0], scaling="log", percent=0.015))

ax[0,0].set_box_aspect(1)
ax[1,0].set_box_aspect(1)

ax[1,0].set_ylim((0,1))

# Set titles, labels, and legends
ax[1,1].set_xlabel('$Shear$ $Strain$ $\u03B3$ / unitless', size=fs)

ax[0,1].set_xlim(adjust_margins((10**1, 10**8), ax[0,1], scaling="log", percent=0.015))
ax[1,1].set_xlim(adjust_margins((10**-1, 10**6), ax[1,1], scaling="log", percent=0.015))

ax[0,0].set_ylim(adjust_margins((-15, 0), ax[0,0], percent=0.015))
ax[0,1].set_ylim(adjust_margins((-15, 0), ax[0,1], percent=0.015))
ax[1,0].set_ylim(adjust_margins((0.1, 0.6), ax[1,0], percent=0.015))
ax[1,1].set_ylim(adjust_margins((0.1, 0.6), ax[1,1], percent=0.015))

ax[0,1].set_box_aspect(1)
ax[1,1].set_box_aspect(1)

ax[0,0].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[0,1].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[1,0].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)
ax[1,1].tick_params(direction="in", top=True, right=True, labelsize=fs, length=4)

ax[1,0].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))
ax[1,1].xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=20))

# Create a legend for files (colors only) at a single point in the figure
handles = [plt.Line2D([0], [0], color=colors[i], lw=2, label = simulation_inputs["Diffusion coefficient"]
[i]) for i in range(len(result_files))]
fig.legend(handles=handles,
          #ncols=2,
          loc=(0.87,0.55),
          title="D / m$^2$s$^{\u2212 1}$",
          fontsize=fs)

plt.savefig(FOLDER+"/MC_evolution_log.png", dpi=300)
plt.show()


#%% METAL-HYDROGEN-SYSTEM PLOTTING - SHEAR HISTOGRAM
fs = 9
cm = 1/2.54

FOLDER = "Example_metal_hydrogen_system"

files = glob(FOLDER+"/*/final_stats.csv")

colors = sns.color_palette("viridis", len(files))

fig, axes = plt.subplots(1, len(files), figsize=( 17.2 *cm, 12 * cm), sharey=True)
fig.subplots_adjust(bottom=0.15, top=0.85, left=0.1, right=0.95)

for i, (file, ax) in enumerate(zip(files, axes)):
    data = pd.read_csv(file)
    shear_hist = []
    for ii in range(len(data)):
        shear_hist.append(np.fromstring(data["Shear Hist Horizontal"].values[ii].strip('[]'), sep=' ', dtype=int))
    shear_hist = np.array(shear_hist)
    lines = np.arange(len(shear_hist[0]))
        
    # Plot the lower bound (value - std) with base color
    ax.plot(shear_hist.mean(axis=0), lines, color=colors[i], alpha=1.0)
    
    ax.fill_betweenx(lines,
                    shear_hist.mean(axis=0) - shear_hist.std(axis=0),
                    shear_hist.mean(axis=0) + shear_hist.std(axis=0),
                    color=colors[i],
                    alpha=0.5)
    
    ax.tick_params(axis='x', labelrotation=45)
    ax.tick_params(labelsize=fs-2)
    #ax.ticklabel_format(axis="x", style="sci", scilimits=(0, 0))
    
    ax.set_ylim(len(shear_hist[0]), 0)
    D = simulation_inputs["Diffusion coefficient"]
    ax.set_title(f"$D$ = {D[i]:.1e}", size=fs, rotation=45)
    
    if i > 0 and i < len(files)-1:
        # Hide the right and top spines
        for spine in ['right', 'left']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(left=False, top=True)
            
    elif i == 0:
        # Hide the right and top spines
        for spine in ['right']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(top=True)
            
    else:
        # Hide the right and top spines
        for spine in ['left']:
            ax.spines[spine].set_visible(False)
            ax.tick_params(top=True, right=True, left=False)
            
            
axes[0].set_ylabel("$Grid$ $Line$ / unitless", size=fs)
axes[round(len(files)/2)].set_xlabel("$Shear$ $Count$ / unitless", size=fs)

#plt.tight_layout()
plt.show()
plt.savefig(FOLDER+"/Shear Step Histogramm.png", dpi=300)

#%% METAL-HYDROGEN-SYSTEM PLOTTING - GENERATE VIDEO

FOLDER = "Example_metal_hydrogen_system"

# Create a custom colormap and swap colors
colors = sns.color_palette("deep", 4)
colors = colors[0], colors[3]  # Swap colors
cmap = LinearSegmentedColormap.from_list("custom_cmap", colors)

# Define the color map for hydrogen grid using violet and white
colors_H2 = [(1, 1, 1), (112/255, 48/255, 160/255)]
cmap_H2 = ListedColormap(colors_H2)

for Dir, Diff in zip(glob(FOLDER+"/*_runs"), simulation_inputs["Diffusion coefficient"]): #.iloc[[1,5,-1]] used for examplary video generation

    # Define paths
    metal_image_list = glob(Dir + "/for_video/run_0/metal_*.npy")
    hydrogen_image_list = glob(Dir + "/for_video/run_0/hydrogen_*.npy")
    
    # Skip video generation if data is not available
    if not metal_image_list:
        continue 
    simulation_results =  Dir + "/simulation_results.csv"
    simulation_results = pd.read_csv(simulation_results)
    simulation_results[simulation_results["Run"] == 1]
    output_folder = Dir
    
    create_metal_hydrogen_video(
        metal_filenames=sorted(metal_image_list),
        hydrogen_filenames=sorted(hydrogen_image_list),
        output_file=  output_folder + f'/output_video_run_{os.path.splitext(metal_image_list[0])[0][-8]}.mp4',
        fps=30,
        cmap_metal=cmap,
        cmap_hydrogen=cmap_H2,
        scale_factor=10,
        stats_df=simulation_results)  # Contains "Step" and "Successful Shear Attempts"
