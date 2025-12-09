"""
This script generates the final 3x2 panel Figure 3 for the paper by combining the
results from the baseline ODE simulations, the myofibril comparison
simulations, and the sensitivity analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import rc

# Import model components needed for the inset plot
from musclex.protocol import DeFreitasProtocol, RegularExercise
from musclex.exercise_model import ExerciseModel


# Import the custom plotting function for the SA plot
try:
    from plot_SA_result import plot_sa_scatter
except ImportError:
    print(
        "Error: Could not import 'plot_sa_scatter'. Make sure 'plot_SA_result.py' is in the scripts directory."
    )
    exit(1)


# --- Matplotlib configuration ---
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


# Define protocols and config path for the inset plot
inset_protocols = {
    "defreitas": DeFreitasProtocol(),
    "everythreedays": RegularExercise(
        N=18,
        exercise_duration=4,
        growth_duration=20 + 24 * 2,
        end_time=9 * 7 * 24,
        initial_rest=7 * 24,
        intensity=0.2,
    ),
    "weekly": RegularExercise(
        N=8,
        exercise_duration=1,
        growth_duration=23 + 24 * 6,
        end_time=9 * 7 * 24,
        initial_rest=7 * 24,
    ),
}
config_path_for_inset = Path(snakemake.input.baseline_config)

ode_states = ["igf1", "akt", "foxo", "mtor"]
state_labels = ["IGF1", "AKT", "FOXO", "mTOR"]

# --- Figure creation ---
fig, axs = plt.subplots(3, 2, figsize=(5, 4.5), constrained_layout=True)
axs = axs.flatten()


# --- Plot 1-4: Baseline ODE states ---
for i, state in enumerate(ode_states):
    ax = axs[i]
    for protocol in plot_order:
        filepath = next(p for p in baseline_result_paths if protocol in p)
        df = pd.read_csv(filepath)
        ax.plot(df.t / (24 * 7), df[state], label=legend_names.get(protocol, protocol))
    ax.set_ylabel(state_labels[i])
    ax.set_xticks([0, 3, 6, 9])

# --- Plot 5: Myofibril comparison ---
ax = axs[4]
for protocol in plot_order:
    filepath = next(p for p in myofibril_result_paths if protocol in p)
    df = pd.read_csv(filepath)
    ax.plot(df.t / (24 * 7), df.z, label=legend_names.get(protocol, protocol))
ax.set_ylabel(r"Myofibrils $N$")
ax.set_ylim(1.0, 1.1)
ax.set_xticks([0, 3, 6, 9])

# --- Plot 6: Sensitivity Analysis ---
# Call function to draw on the last subplot
plot_sa_scatter(axs[5], sa_csv_path)
# axs[5].set_title("Sobol Sensitivity Analysis")

# --- Final styling and legend ---
# Set common x-axis labels
for i in [0, 1, 2, 3, 4]:  # Add labels to all plots
    axs[i].set_xlabel("Time (weeks)")

# Common styling for all axes
for ax in axs:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(top=False, right=False, which="both")
    ax.minorticks_off()

# Create a single, shared legend for the top plots
lines, labels = axs[0].get_legend_handles_labels()
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
plt.savefig(Path(output_figure_path).with_suffix(".png"), dpi=300, bbox_inches="tight") # png

print(f"Final figure saved to {output_figure_path}")
