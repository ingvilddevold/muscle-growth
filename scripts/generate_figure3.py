"""
This script generates the final multipanel Figure 3 for the paper by combining the
results from the baseline ODE simulations, the myofibril comparison
simulations, and the sensitivity analysis.
"""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt


# Import the custom plotting function for the SA plot
try:
    from plot_SA_result import plot_sa_scatter
except ImportError:
    print(
        "Error: Could not import 'plot_sa_scatter'. Make sure 'plot_SA_result.py' is in the scripts directory."
    )
    exit(1)


# --- Matplotlib configuration ---
from matplotlib import rc
import scienceplots

plt.style.use("science")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 8
plt.rcParams["svg.fonttype"] = "none"

rc("text", usetex=False)

# --- Snakemake integration ---
baseline_result_paths = snakemake.input.protocol_results
myofibril_result_paths = snakemake.input.myofibril_results
sa_csv_path = snakemake.input.sa_results
output_figure_path = snakemake.output[0]
protocol_names = snakemake.params.protocols


# --- Setup ---
legend_names = {
    "defreitas": "MWF",
    "weekly": "Weekly",
    "everythreedays": "Every three days",
}

# Define a specific plotting order to ensure consistent default colors:
plot_order = ["defreitas", "everythreedays", "weekly"]
# Filter the plot order to only include protocols actually run by the workflow
plot_order = [p for p in plot_order if p in protocol_names]

ode_states = ["igf1", "akt", "foxo", "mtor"]
state_labels = ["IGF1", "AKT", "FOXO", "mTOR"]

# --- Figure creation ---
# 4x2 grid
fig, axs = plt.subplots(
    4,
    2,
    figsize=(5, 4.5),
    gridspec_kw={"height_ratios": [0.4, 1, 1, 0.9]},
    constrained_layout=True,
)
axs = axs.flatten()

# --- Map axes ---
# Hide the default top row axes
axs[0].axis("off")
axs[1].axis("off")

# Create narrower inset axes
# [x0, y0, width, height]
ax_zoom_igf1 = axs[0].inset_axes([0.1, 0.0, 0.5, 1.0])
ax_zoom_akt = axs[1].inset_axes([0.1, 0.0, 0.5, 1.0])

zoom_axes = {"igf1": ax_zoom_igf1, "akt": ax_zoom_akt}
main_axes = {"igf1": axs[2], "akt": axs[3], "foxo": axs[4], "mtor": axs[5]}

# --- Plot 1-6: Baseline ODE states and Zoom-ins ---
for i, state in enumerate(ode_states):
    ax_main = main_axes[state]
    ax_zoom = zoom_axes.get(state)

    for j, protocol in enumerate(plot_order):
        filepath = next(p for p in baseline_result_paths if protocol in p)
        df = pd.read_csv(filepath)

        # Main plot (x-axis in weeks)
        ax_main.plot(
            df.t / (24 * 7), df[state], label=legend_names.get(protocol, protocol)
        )

        # Zoom plot data
        ls = ["solid", "solid", "dashed"]
        if ax_zoom is not None:
            ax_zoom.plot(df.t / (24 * 7), df[state], ls=ls[j])

    # Standard styling for main axes
    ax_main.set_ylabel(state_labels[i])
    ax_main.set_xticks([0, 3, 6, 9])

    # Configure the zoom window row
    if ax_zoom is not None:
        t_start_d = 7
        t_end_d = 8

        # Set limits in weeks
        ax_zoom.set_xlim(t_start_d / 7, t_end_d / 7)

        # ax_zoom.set_ylabel(state_labels[i])

        # Remove x-ticks completely
        ax_zoom.set_xticks([])
        ax_zoom.set_yticks([])
        ax_zoom.set_xlabel("")

        # Add custom text inside the frame
        ax_zoom.text(
            0.25,
            0.75,
            "Day 7 to 8",
            transform=ax_zoom.transAxes,
            ha="center",
            va="bottom",
            fontsize=6,
        )

# --- Plot 7: Myofibril comparison (Row 4, Left) ---
ax = axs[6]
for protocol in plot_order:
    filepath = next(p for p in myofibril_result_paths if protocol in p)
    df = pd.read_csv(filepath)
    ax.plot(df.t / (24 * 7), df.z, label=legend_names.get(protocol, protocol))
ax.set_ylabel(r"Myofibrils $N$")
ax.set_ylim(1.0, 1.1)
ax.set_xticks([0, 3, 6, 9])

# --- Plot 8: Sensitivity Analysis (Row 4, Right) ---
plot_sa_scatter(axs[7], sa_csv_path)

# --- Final styling and legend ---
# Set common x-axis labels for the non-zoom rows
for idx in [2, 3, 4, 5, 6]:
    axs[idx].set_xlabel("Time (weeks)")

# Common styling for all axes (including zoom-ins)
all_axes = list(axs) + list(zoom_axes.values())
for ax in all_axes:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.minorticks_off()

# Create a single, shared legend using the top-most main plot
lines, labels = axs[2].get_legend_handles_labels()
fig.legend(
    lines,
    labels,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.05),
    ncol=3,
    frameon=False,
)

# --- Save figure ---
plt.savefig(output_figure_path, dpi=300, bbox_inches="tight")  # svg
plt.savefig(
    Path(output_figure_path).with_suffix(".png"), dpi=300, bbox_inches="tight"
)  # png

print(f"Final figure saved to {output_figure_path}")
