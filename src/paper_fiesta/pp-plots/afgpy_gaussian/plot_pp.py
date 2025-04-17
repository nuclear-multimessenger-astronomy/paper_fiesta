import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats

def boundary(p_array, n: int):
    a = [stats.binom.ppf(0.025, n, p) for p in p_array]
    b = [stats.binom.ppf(0.975, n, p) for p in p_array]
    return np.array(a)/n, np.array(b)/n

def pp_plot(ax, quantiles, p_array):
    lower_boundary, upper_boundary = boundary(p_array, quantiles.shape[0])
    #quantiles = self.recovered_quantiles[p] #2*jnp.minimum(self.recovered_quantiles[p], 1-self.recovered_quantiles[p])
    ax.hist(quantiles, density = True, cumulative = True, histtype="step", bins=p_array, color = "blue")
    ax.plot(p_array, p_array, linestyle= "dotted", color = "lightgrey")
    ax.fill_between(p_array, lower_boundary, upper_boundary, color = "lightgrey", alpha = 0.3)
    
    ax.set_xlabel("posterior quantile $p$")
    ax.set_ylabel("CDF")
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))



quantiles = np.loadtxt("./outdir/quantiles.txt")
parameter_names = ["$\\iota$", "$\\log_{10}(E_0)$", "$\\theta_{\\mathrm{c}}$", "$\\alpha_{\\mathrm{w}}$", "$\\log_{10}(n_{\mathrm{ism}})$", "$p$", "$\\log_{10}(\\epsilon_e)$", "$\\log_{10}(\\epsilon_B)$"]



fig, ax = plt.subplots(1, 1, figsize = (8, 5))
fig.subplots_adjust(hspace = 0.4, wspace = 0.1, top = 0.98, bottom = 0.1, left = 0.08, right = 0.98)
p_array = np.linspace(0, 1, 50)
pp_plot(ax, quantiles.flatten(), p_array)
fig.savefig("./outdir/pp_plot_total.pdf", dpi = 250)


fig, ax = plt.subplots(8, 1, figsize = (8, 32))
fig.subplots_adjust(hspace = 0.4, wspace = 0.1, top = 0.98, bottom = 0.05, left = 0.08, right = 0.98)
p_array = np.linspace(0, 1, 50)
for j, cax in enumerate(ax):
    pp_plot(cax, quantiles[:, j], p_array)
    cax.text(0.1, 0.8, parameter_names[j])
fig.savefig("./outdir/pp_plot_parameter.pdf", dpi=250)