"""
Run harmonic to compute the evidences

In case a multimodal base is needed, check out this branch: https://github.com/astro-informatics/harmonic/tree/multimodal_base
"""

import os
import numpy as np 
import matplotlib.pyplot as plt
import time
import corner
import json
import argparse

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import harmonic as hm

params = {"axes.grid": False,
        "text.usetex" : False,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        # "font.serif" : ["Computer Modern Serif"],
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "axes.labelsize": 16,
        "legend.fontsize": 16,
        "legend.title_fontsize": 16,
        "figure.titlesize": 16}

plt.rcParams.update(params)

path_fiesta_run = "../injections/GRB/afgpy_gaussian/outdir_fiesta/results_production.npz"

# # Load it:
# data_fiesta = np.load(path_fiesta_run)
# chains = data_fiesta["chains"]
# log_prob = data_fiesta["log_prob"]

# # Reshape
# n_chains, n_steps, n_dim = chains.shape
# chains = chains.reshape(n_chains * n_steps, n_dim)
# log_prob = log_prob.flatten()

# Compare to the NMMA run (using the surrogate for the comparison)
paths_dict = {"afgpy": "../injections/GRB/afgpy_gaussian/outdir_nmma/injection_gaussian_result.json",
              "fiesta in NMMA": "../injections/GRB/afgpy_gaussian/outdir_nmma_x_fiesta/injection_gaussian_result.json",
              }

for name, path in paths_dict.items():
    with open(path, "r") as f:
        data_nmma = json.load(f)
        log_evidence = data_nmma["log_evidence"]
        sampling_time = data_nmma["sampling_time"]
        
    print(f"Log evidence for: {name}: {log_evidence:.3f}")
    print(f"Sampling time for: {name}: {sampling_time:.3f}")