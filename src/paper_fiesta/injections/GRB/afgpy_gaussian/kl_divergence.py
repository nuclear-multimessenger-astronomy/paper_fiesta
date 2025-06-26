import os

import numpy as np
from scipy.stats import gaussian_kde

from fiesta.utils import load_event_data

for j in range(1,6):
    if j==1:
        directory = f"."
    else:
        directory = f"../afgpy_gaussian_{j}"

    posterior_fiesta = np.loadtxt(os.path.join(directory, "outdir_nmma_x_fiesta/injection_gaussian_posterior_samples.dat"), skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    posterior_nmma = np.loadtxt(os.path.join(directory, "outdir_nmma/injection_gaussian_posterior_samples.dat"), skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])
    
    #ind = np.random.choice(len(posterior_fiesta), size=700_000, replace=False)
    
    nmma_density = gaussian_kde(posterior_nmma.T)
    fiesta_density = gaussian_kde(posterior_fiesta[:].T)
    
    entropy = -  np.mean(np.log(nmma_density(posterior_nmma.T)))
    cross_entropy = - np.mean(np.log(fiesta_density(posterior_nmma.T)))


    data = load_event_data(os.path.join(directory, "data/injection_afterglowpy_gaussian.dat"))

    max_sigma = 0.
    for key in data.keys():
        max_sigma = max(max_sigma, np.max(data[key][:,2]))

    upper_bound = 75/2 * 0.1**2 / max_sigma**2

    kl_divergence = - entropy + cross_entropy
    
    print(directory, kl_divergence, upper_bound / kl_divergence)
