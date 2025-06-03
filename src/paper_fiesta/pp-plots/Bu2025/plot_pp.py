import matplotlib.pyplot as plt
import numpy as np

import scipy.stats as stats

from fiesta.inference.prior import Uniform

plt.rcParams.update({"text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"]})

def boundary(p_array, n: int):
    a = [stats.binom.ppf(0.025, n, p) for p in p_array]
    b = [stats.binom.ppf(0.975, n, p) for p in p_array]
    return np.array(a)/n, np.array(b)/n

def pp_plot(ax, quantiles, p_array, double=False, color="blue"):
    lower_boundary, upper_boundary = boundary(p_array, quantiles.shape[0])
    if double:
        quantiles = 2*np.minimum(quantiles, 1- quantiles)
    ax.hist(quantiles, density = True, cumulative=True, histtype="step", bins=p_array, color=color)
    ax.plot(p_array, p_array, linestyle= "dotted", color = "lightgrey")
    ax.fill_between(p_array, lower_boundary, upper_boundary, color = "lightgrey", alpha = 0.3)
    
    ax.set_xlabel("posterior quantile $p$", fontsize=15)
    ax.set_ylabel("CDF", fontsize=15)
    ax.tick_params(axis="both", which="major", labelsize=12)
    
    ax.set_xlim((0,1))
    ax.set_ylim((0,1))


inclination_EM = Uniform(xmin=0.0, xmax=np.pi/2, naming=['inclination_EM'])
log10_mej_dyn = Uniform(xmin=-3.0, xmax=-1.30, naming=["log10_mej_dyn"])
v_ej_dyn = Uniform(xmin=0.12, xmax=0.28, naming=["v_ej_dyn"])
Ye_dyn = Uniform(xmin=0.15, xmax=0.35, naming=["Ye_dyn"])
log10_mej_wind = Uniform(xmin=-2.0, xmax=-0.89, naming=["log10_mej_wind"])
v_ej_wind = Uniform(xmin=0.05, xmax=0.15, naming=["v_ej_wind"])
Ye_wind = Uniform(xmin=0.2, xmax=0.4, naming=["Ye_wind"])
sys_err = Uniform(xmin=0.3, xmax=1.0, naming=["sys_err"])

prior_list = [inclination_EM, 
              log10_mej_dyn,
              v_ej_dyn,
              Ye_dyn,
              log10_mej_wind,
              v_ej_wind,
              Ye_wind,
              sys_err]


quantiles = np.loadtxt("./outdir/quantiles.txt")
params = np.loadtxt("./outdir/params.txt")
parameter_names = ["$\\iota$", "$\log_{10}(m_{\\mathrm{ej, dyn}})$", "$\\bar{v}_{\\mathrm{ej, dyn}}$", "$\\bar{Y}_{e, \\mathrm{dyn}}$", "$\log_{10}(m_{\\mathrm{ej, wind}})$", "$\\bar{v}_{\\mathrm{ej, wind}}$", "$Y_{e, \\mathrm{wind}}$"]
double_list = [True, True, False, False, False, False, True]

fig, ax = plt.subplots(1, 1, figsize = (8, 5))
fig.subplots_adjust(hspace = 0.4, wspace = 0.1, top = 0.98, bottom = 0.1, left = 0.08, right = 0.98)
p_array = np.linspace(0, 1, 50)
pp_plot(ax, quantiles.flatten(), p_array)
fig.savefig("./outdir/pp_plot_total.pdf", dpi = 250)


fig, ax = plt.subplots(4, 2, figsize = (8, 14.4))
fig.subplots_adjust(hspace = 0.25, wspace = 0.3, top = 0.98, bottom = 0.05, left = 0.1, right = 0.98)
p_array = np.linspace(0, 1, 50)
for j, cax in enumerate(ax.flatten()[:-1]):
    
    # when the injected value is at the edge of the prior, we should exclude the injected value because the quantile will be 0 or 1
    mask = (params[:,j]> prior_list[j].xmin) & (params[:,j] <  prior_list[j].xmax)

    pp_plot(cax, quantiles[mask, j], p_array, double=double_list[j], color="purple")
    cax.text(0.1, 0.8, parameter_names[j], fontsize=15)


handles = []
for c in ["purple"]:
    handle = plt.plot([],[], color=c)[0]
    handles.append(handle)

ax[3,1].set_axis_off()
ax[0,0].legend(handles=handles, labels=["Surrogate \\textsc{possis}"], fontsize=12, fancybox=False, framealpha=1)


fig.savefig("./outdir/pp_plot_KN.pdf", dpi=250)
