import numpy as np
from scipy.stats import gaussian_kde


posterior_fiesta = np.load("./outdir_fiesta/results_production.npz")["chains"].reshape(-1, 9)
posterior_nmma = np.loadtxt("./outdir_nmma/injection_gaussian_posterior_samples.dat", skiprows=1, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8])

#ind = np.random.choice(len(posterior_fiesta), size=700_000, replace=False)

nmma_density = gaussian_kde(posterior_nmma.T)
fiesta_density = gaussian_kde(posterior_fiesta[:].T)

entropy = -  np.mean(np.log(nmma_density(posterior_nmma.T)))
cross_entropy = - np.mean(np.log(fiesta_density(posterior_nmma.T)))

print(- entropy + cross_entropy)
breakpoint()
