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
    ess = []

    for d in D:
        mask = data[:,0]==d
        total_time.append(np.mean(data[mask, 1]))
        compile_time.append(np.mean(data[mask,2]))
        ess.append(np.mean(data[mask, 3]))

    return D, np.array(total_time), np.array(compile_time), np.array(ess)

def plot_performance(ax, filename, color, label):
    D, total_time, compile_time, ess = read_timing_file(filename)
    ax.plot(D, (total_time-compile_time)/ess, linestyle="dashed", color="grey", alpha=0.5)
    ax.scatter(D, (total_time-compile_time)/ess, c=color, marker="s", label=label)

def plot_performance_5000(ax, filename, color, label):
    D, total_time, compile_time, ess = read_timing_file(filename)
    ax.plot(D, 5000*(total_time-compile_time)/ess, linestyle="dashed", color="grey", alpha=0.5)
    ax.scatter(D, 5e3*(total_time-compile_time)/ess, c=color, marker="s", label=label)


fig, ax = plt.subplots(1, 1, figsize=(4, 4.8*2/3))
fig.subplots_adjust(bottom=0.12, left=0.1, top=0.98, right=0.98)

ax2 = ax.twinx()


plot_performance(ax, "./outdir/RTX_6000_timing.txt", color="blue", label="NVIDIA RTX 6000")
plot_performance(ax, "./outdir/H100_timing.txt", color="red", label="NVIDIA H100")


plot_performance_5000(ax2, "./outdir/RTX_6000_timing.txt", color="blue", label="NVIDIA RTX 6000")
plot_performance_5000(ax2, "./outdir/H100_timing.txt", color="red", label="NVIDIA H100")



ax.set_xlabel("Number of sampling parameters", fontsize=14)
ax.set_ylabel("Sampling time/ESS in s", fontsize=14)
ax2.set_ylabel("Sampling time(5000 ESS) in s", fontsize=14)
#ax.set(xscale="log", yscale="log")

ax.set_xticks([10, 20, 40, 80], [10, 20, 40, 80], fontsize=12)
ax.set_xticks([], [], fontsize=12, minor=True)

#ax.set_yticks([50, 70, 100, 140], [50, 70, 100, 140], fontsize=12)'
#ax.set_yticks(range(50, 250, 50), range(50, 250, 50), fontsize=12)
#x.set_yticks([], [], fontsize=12, minor=True)
ax.legend(framealpha=1, fancybox=False, fontsize=14)

fig.savefig("./outdir/performance_plot.pdf", dpi=250, bbox_inches="tight")