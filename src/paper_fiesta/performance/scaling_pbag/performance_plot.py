import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"]})


fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))

D, total_time, compile_time = np.loadtxt("./outdir/Quadro_RTX_6000_timing.txt", unpack=True)
ax.plot(D, total_time, linestyle="dashed", color="grey", alpha=0.5)
ax.scatter(D, total_time, c="blue", marker="s", label="NVIDIA RTX 6000")



ax.set_xlabel("Number of sampling parameters", fontsize=15)
ax.set_ylabel("Runtime in s", fontsize=15)
ax.set(xscale="log", yscale="log")

ax.set_xticks([10, 20, 40], [10, 20, 40], fontsize=12)
ax.set_yticks([70, 80, 90, 100], [70, 80, 90, 100], fontsize=12)

ax.set_xticks([], [], fontsize=12, minor=True)
ax.set_yticks([], [], fontsize=12, minor=True)
ax.legend(framealpha=1, fancybox=False, fontsize=15)

fig.savefig("./outdir/performance_plot.pdf", dpi=250)