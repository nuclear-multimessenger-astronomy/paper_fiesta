import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats

plt.rcParams.update({"text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"]})

def boundary(p_array, n: int, alpha=0.05):
    a = [stats.binom.ppf(alpha/2, n, p) for p in p_array]
    b = [stats.binom.ppf(1-alpha/2, n, p) for p in p_array]
    return np.array(a)/n, np.array(b)/n

def pp_plot(ax, quantiles, p_array, double=False, color="blue"):
    if double:
        quantiles = 2*np.minimum(quantiles, 1- quantiles)
    #quantiles = self.recovered_quantiles[p] #2*jnp.minimum(self.recovered_quantiles[p], 1-self.recovered_quantiles[p])
    ax.hist(quantiles, density = True, cumulative = True, histtype="step", bins=p_array, color=color)
    ax.plot(p_array, p_array, linestyle= "dotted", color = "lightgrey")

    lower_boundary, upper_boundary = boundary(p_array, quantiles.shape[0])
    ax.fill_between(p_array, lower_boundary, upper_boundary, color="lightgrey", alpha=0.6)

    lower_boundary, upper_boundary = boundary(p_array, quantiles.shape[0], alpha=0.32)
    ax.fill_between(p_array, lower_boundary, upper_boundary, color="lightgrey", alpha=0.8)

    lower_boundary, upper_boundary = boundary(p_array, quantiles.shape[0], alpha=0.003)
    ax.fill_between(p_array, lower_boundary, upper_boundary, color="lightgrey", alpha=0.4)
    
    ax.set_xlabel("posterior quantile $p$", fontsize=15)
    ax.set_ylabel("CDF", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))



quantiles = np.loadtxt("./outdir/quantiles.txt")
quantiles_pbag_surrogate = np.loadtxt("../pbag_gaussian_surrogate/outdir/quantiles.txt")
quantiles_afgpy = np.loadtxt("../afgpy_gaussian/outdir/quantiles.txt")
quantiles_afgpy_surrogate = np.loadtxt("../afgpy_gaussian_surrogate/outdir/quantiles.txt")
parameter_names = ["$\\iota$", "$\\log_{10}(E_0)$", "$\\theta_{\\mathrm{c}}$", "$\\alpha_{\\mathrm{w}}$", "$\\log_{10}(n_{\mathrm{ism}})$", "$p$", "$\\log_{10}(\\epsilon_e)$", "$\\log_{10}(\\epsilon_B)$", "$\\Gamma_0$"]
double_list = [False, False, False, True, False, False, False, False, False]

fig, ax = plt.subplots(1, 1, figsize = (8, 5))
fig.subplots_adjust(hspace = 0.4, wspace = 0.1, top = 0.98, bottom = 0.1, left = 0.08, right = 0.98)
p_array = np.linspace(0, 1, 50)
pp_plot(ax, quantiles.flatten(), p_array)
fig.savefig("./outdir/pp_plot_total.pdf", dpi = 250)


fig, ax = plt.subplots(5, 2, figsize = (8, 18))
fig.subplots_adjust(hspace = 0.25, wspace = 0.3, top = 0.98, bottom = 0.05, left = 0.1, right = 0.98)
p_array = np.linspace(0, 1, 50)
for j, cax in enumerate(ax.flatten()[:9]):
    pp_plot(cax, quantiles[:, j], p_array, double=double_list[j], color="magenta")
    cax.text(0.1, 0.8, parameter_names[j], fontsize=18)

for j, cax in enumerate(ax.flatten()[:8]):

    if double_list[j]:
        cax.hist(2*np.minimum(quantiles_afgpy_surrogate[:, j], 1-quantiles_afgpy[:,j]), density=True, cumulative=True, histtype="step", bins=p_array, color="deepskyblue", linestyle="dashed")
        cax.hist(2*np.minimum(quantiles_afgpy[:, j], 1-quantiles_afgpy[:,j]), density=True, cumulative=True, histtype="step", bins=p_array, color="deepskyblue")
        cax.hist(2*np.minimum(quantiles_pbag_surrogate[:, j], 1-quantiles_afgpy[:,j]), density=True, cumulative=True, histtype="step", bins=p_array, color="magenta", linestyle="dashed")
    else:
        cax.hist(quantiles_afgpy[:, j], density=True, cumulative=True, histtype="step", bins=p_array, color = "deepskyblue")
        cax.hist(quantiles_afgpy_surrogate[:, j], density=True, cumulative=True, histtype="step", bins=p_array, color = "deepskyblue", linestyle="dashed")
        cax.hist(quantiles_pbag_surrogate[:, j], density=True, cumulative=True, histtype="step", bins=p_array, color = "magenta", linestyle="dashed")

ax[-1,0].hist(quantiles_pbag_surrogate[:, -1], density=True, cumulative=True, histtype="step", bins=p_array, color = "magenta", linestyle="dashed")

handles = []
for c in ["deepskyblue", "magenta"]:
    handle = plt.plot([],[], color=c)[0]
    handles.append(handle)

ax[4,1].set_axis_off()
ax[-1, -1].legend(handles=handles, labels=["\\textsc{afterglowpy}", "\\textsc{pyblastafterglow}"], fontsize=15, fancybox=False, framealpha=1, loc="lower center")


fig.savefig("./outdir/pp_plot_GRB.pdf", dpi=250)
