import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import numpy as np
from pathlib import Path
from typing_extensions import Annotated
from matplotlib.ticker import MaxNLocator

# --- Matplotlib Styling ---
import scienceplots

plt.style.use("science")
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 8
plt.rcParams["svg.fonttype"] = "none"
rc("text", usetex=False)

# Re-enable the axis offset for small numbers
plt.rcParams["axes.formatter.useoffset"] = True

# --- Parameter Name Mapping ---
# 1->I (IGF), 2->A (AKT), 3->F (FOXO), 4->M (mTOR)
PARAM_LABEL_MAP = {
    "a10": "$a_{I0}$",
    "a20": "$a_{A0}$",
    "a3": "$a_{F}$",
    "a4": "$a_{M}$",
    "b1": "$b_{I}$",
    "b2": "$b_{A}$",
    "b3": "$b_{F}$",
    "b4": "$b_{M}$",
    "c21": "$c_{AI}$",
    "c32": "$c_{FA}$",
    "c42": "$c_{MA}$",
    "c43": "$c_{MF}$",
}


@typer.run
def main(
    param_csv: Annotated[
        Path, typer.Option(help="Path to the spatial_parameters.csv file.")
    ],
    config_file: Annotated[
        Path, typer.Option(help="Path to the exercise_eq_reduced_k1.yml config file.")
    ],
    output_file: Annotated[
        Path, typer.Option(help="Path to save the output PNG file.")
    ],
):
    """
    Plots histograms of spatially varying parameters against their baseline values.
    """

    # 1. Load baseline parameters
    with open(config_file) as f:
        conf = yaml.safe_load(f)
    baseline_params = conf["exercise_parameters"]

    # 2. Load spatial parameter data
    df = pd.read_csv(param_csv)

    parameters_to_plot = [col for col in df.columns if col in baseline_params]
    if not parameters_to_plot:
        print("Error: No matching parameters found between CSV and config file.")
        raise typer.Exit(code=1)

    # 3. Set up the plot grid
    n_params = len(parameters_to_plot)
    ncols = 4  # 4 plots per row
    nrows = int(np.ceil(n_params / ncols))

    fig, axes = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=(ncols * 2.5, nrows * 2), squeeze=False
    )

    print(f"Plotting {n_params} parameter distributions...")

    # 4. Create each histogram plot
    for i, param_name in enumerate(parameters_to_plot):
        ax = axes.flat[i]

        data = df[param_name]
        baseline_val = baseline_params.get(param_name)
        plot_title = PARAM_LABEL_MAP.get(param_name, param_name) + f" (baseline {baseline_val:.2g})"
        ax.set_title(plot_title, fontsize=10, pad=2)

        # Plot histogram and KDE
        sns.histplot(data, ax=ax, bins=30, stat="count", kde=True, alpha=0.7, color="gray")

        # Plot baseline value as a vertical line
        if baseline_val is not None:
            ax.axvline(
                baseline_val,
                color="black",
                linestyle="--",
                linewidth=1,
            )

        # Apply styling
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.minorticks_off()

        # --- FIX: Limit ticks and style the offset ---
        # 1. Limit the number of x-ticks to prevent overlap
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="both"))
        # 2. Make the offset text (e.g., "+1.2e-3") small
        ax.xaxis.get_offset_text().set_fontsize("x-small")

    # 5. Hide any unused subplots
    for i in range(n_params, len(axes.flat)):
        axes.flat[i].set_visible(False)

    plt.tight_layout()

    # Save the figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")

    print(f"Successfully saved parameter distribution plot to {output_file}")
