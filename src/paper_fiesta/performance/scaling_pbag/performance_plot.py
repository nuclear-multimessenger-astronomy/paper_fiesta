import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"text.usetex" : True,
        "font.family" : "serif",
        "ytick.color" : "black",
        "xtick.color" : "black",
        "axes.labelcolor" : "black",
        "axes.edgecolor" : "black",
        "font.serif" : ["Computer Modern Serif"]})


def read_timing_file(filename):
    data = np.loadtxt(filename)
    D = np.sort(np.unique(data[:,0]))
    
    total_time = []
    compile_time = []

    for d in D:
        mask = data[:,0]==d
        total_time.append(np.mean(data[mask, 1]))
        compile_time.append(np.mean(data[mask,2]))

    return D, np.array(total_time), np.array(compile_time)


fig, ax = plt.subplots(1, 1, figsize=(6, 4.8))
fig.subplots_adjust(bottom=0.12, left=0.1, top=0.98, right=0.98)

D, total_time, compile_time = read_timing_file("./outdir/RTX_6000_timing.txt")
ax.plot(D, total_time-compile_time, linestyle="dashed", color="grey", alpha=0.5)
ax.scatter(D, total_time-compile_time, c="blue", marker="s", label="NVIDIA RTX 6000")

D, total_time, compile_time = read_timing_file("./outdir/H100_timing.txt")
ax.plot(D, total_time-compile_time, linestyle="dashed", color="grey", alpha=0.5)
ax.scatter(D, total_time-compile_time, c="red", marker="s", label="NVIDIA H100")



ax.set_xlabel("Number of sampling parameters", fontsize=15)
ax.set_ylabel("Runtime in s", fontsize=15)
#ax.set(xscale="log", yscale="log")

ax.set_xticks([10, 20, 40, 80], [10, 20, 40, 80], fontsize=12)
#ax.set_yticks([50, 70, 100, 140], [50, 70, 100, 140], fontsize=12)'
ax.set_yticks(range(50, 250, 50), range(50, 250, 50), fontsize=12)

ax.set_xticks([], [], fontsize=12, minor=True)
ax.set_yticks([], [], fontsize=12, minor=True)
ax.legend(framealpha=1, fancybox=False, fontsize=15)

fig.savefig("./outdir/performance_plot.pdf", dpi=250)