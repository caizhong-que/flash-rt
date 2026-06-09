import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def plot_all(re):
    rcParams["xtick.labelsize"] = 11
    rcParams["ytick.labelsize"] = 11
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["axes.grid"] = True
    rcParams["grid.alpha"] = 0.3
    rcParams["grid.linestyle"] = "--"
    rcParams["axes.linewidth"] = 0.8
    rcParams["lines.linewidth"] = 1.8
    rcParams["lines.markersize"] = 6
    rcParams["lines.markeredgewidth"] = 0.8

    batch_sizes = np.array(re["Batch_size"]) / 10000
    time_s = re["Time(s)"]
    ram_mb = re["RAM-Max(MB)"]
    gpu_mb = re["GPU-max(MB)"]
    cpu_percent = re["CPU(%)"]
    precision = re["P"]
    recall = re["R"]
    f1 = re["F1"]
    hitrate = re["Hit"]

    colors = {
        "P": "#0173B2",
        "R": "#DE8F05",
        "F1": "#029E73",
        "Hit": "#D55E00",
        "RAM": "#CC78BC",
        "GPU": "#949494",
        "CPU": "#56B4E9",
        "Time": "#CA9161",
    }

    markers = {
        "P": "o",
        "R": "s",
        "F1": "^",
        "Hit": "D",
        "RAM": "v",
        "GPU": "P",
        "CPU": "X",
        "Time": "*",
    }

    fig1, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(batch_sizes, precision, marker=markers["P"], color=colors["P"], label="Precision")
    ax1.plot(batch_sizes, recall, marker=markers["R"], color=colors["R"], label="Recall")
    ax1.plot(batch_sizes, f1, marker=markers["F1"], color=colors["F1"], label="F1")
    ax1.set_xlabel(r"Batch Size ($\times 10^4$)", fontweight="bold")
    ax1.set_ylabel("Score", fontweight="bold")
    ax1.set_xticks([0, 5, 10, 15, 20, 25])
    ax1.set_xlim(0, 26)
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    plt.tight_layout()
    plt.savefig("fig1_performance_metrics.pdf", bbox_inches="tight", format="pdf")
    plt.savefig("fig1_performance_metrics.png", bbox_inches="tight")
    plt.show()

    fig2, ax2 = plt.subplots(figsize=(8, 5))
    ax2.plot(batch_sizes, ram_mb, marker=markers["RAM"], color=colors["RAM"], label="RAM Usage")
    ax2.plot(batch_sizes, gpu_mb, marker=markers["GPU"], color=colors["GPU"], label="GPU Memory")
    ax2.set_xlabel(r"Batch Size ($\times 10^4$)", fontweight="bold")
    ax2.set_ylabel("Memory Usage (MB)", fontweight="bold")
    ax2.set_xticks([0, 5, 10, 15, 20, 25])
    ax2.set_xlim(0, 26)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(batch_sizes, cpu_percent, marker=markers["CPU"], color=colors["CPU"], label="CPU Usage", linestyle="--")
    ax2_twin.set_ylabel("CPU Usage (%)", fontweight="bold", color=colors["CPU"])
    ax2_twin.tick_params(axis="y", labelcolor=colors["CPU"])

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=True, fancybox=False, edgecolor="black")

    plt.tight_layout()
    plt.savefig("fig2_resource_overhead.pdf", bbox_inches="tight", format="pdf")
    plt.savefig("fig2_resource_overhead.png", bbox_inches="tight")
    plt.show()

    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.plot(batch_sizes, time_s, marker=markers["Time"], color=colors["Time"], linewidth=2.2, markersize=7, label="Processing Time")
    ax3.fill_between(batch_sizes, time_s, alpha=0.15, color=colors["Time"])
    ax3.set_xlabel(r"Batch Size ($\times 10^4$)", fontweight="bold")
    ax3.set_ylabel("Time (s)", fontweight="bold")
    ax3.set_xticks([0, 5, 10, 15, 20, 25])
    ax3.set_xlim(0, 26)
    ax3.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")

    for i, (x, y) in enumerate(zip(batch_sizes, time_s)):
        if i % 2 == 0:
            ax3.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("fig3_processing_time.pdf", bbox_inches="tight", format="pdf")
    plt.savefig("fig3_processing_time.png", bbox_inches="tight")
    plt.show()

    fig4, ax4 = plt.subplots(figsize=(8, 5))
    ax4.plot(batch_sizes, hitrate, marker=markers["Hit"], color=colors["Hit"], linewidth=2.2, markersize=7, label="Hit Rate")
    ax4.set_xlabel(r"Batch Size ($\times 10^4$)", fontweight="bold")
    ax4.set_ylabel("Hit Rate", fontweight="bold")
    ax4.set_xticks([0, 5, 10, 15, 20, 25])
    ax4.set_xlim(0, 26)
    ax4.set_ylim(0, 1.05)
    ax4.legend(loc="best", frameon=True, fancybox=False, edgecolor="black")
    ax4.grid(True, alpha=0.3)

    for i, (x, y) in enumerate(zip(batch_sizes, hitrate)):
        if i % 2 == 0:
            ax4.annotate(f"{y:.2f}", (x, y), textcoords="offset points", xytext=(0, 10), ha="center", fontsize=9)

    plt.tight_layout()
    plt.savefig("fig4_hit_rate.pdf", bbox_inches="tight", format="pdf")
    plt.savefig("fig4_hit_rate.png", bbox_inches="tight")
    plt.show()

    print("\n% LaTeX Table for Paper")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\caption{Performance and Resource Overhead under Different Batch Sizes}")
    print("\\label{tab:batch_size}")
    print("\\begin{tabular}{ccccccccc}")
    print("\\toprule")
    print("Batch Size & Time (s) & RAM (MB) & GPU (MB) & Precision & Recall & F1 & Hit Rate & CPU (\\\\%) \\\\")
    print("\\midrule")