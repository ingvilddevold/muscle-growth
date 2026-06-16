import typer
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import List

# Matplotlib styling
import scienceplots
plt.style.use("science")
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 7
plt.rcParams['svg.fonttype'] = 'none'

rc("text", usetex=False)

# Create a Typer application
app = typer.Typer()

@app.command()
def main(
    csv_files: List[Path] = typer.Argument(
        ...,
        help="One or more paths to the growth_results.csv files.",
    ),
    meshes: str = typer.Option(
        ...,
        "--mesh-names",
        help="Comma-separated list of mesh names, in the same order as csv-files.",
        rich_help_panel="Input/Output"
    ),
    output_file: Path = typer.Option(
        ...,
        "--output-file",
        help="Path to save the output PNG figure.",
        rich_help_panel="Input/Output"
    )
):
    """
    Generates a 2-panel figure comparing CSA and Volume over time
    for different mesh refinement levels to evaluate spatial convergence.
    """
    mesh_list = meshes.split(',')

    # --- Plotting Setup ---
    fig, axes = plt.subplots(1, 2, figsize=(4, 1.5))
    
    # Use a sequential colormap to represent refinement levels (e.g., light to dark)
    colors = plt.cm.viridis(np.linspace(0, 0.9, len(mesh_list)))
    color_map = {mesh: colors[i] for i, mesh in enumerate(mesh_list)}
    linestyles = ["-", "--", ":"]

    # --- Process and Plot Data ---
    for i, csv_file in enumerate(csv_files):
        # Identify which mesh this CSV belongs to based on the file path
        mesh_name = next((m for m in mesh_list if m in str(csv_file)), None)
        if not mesh_name:
            print(f"Warning: Could not determine mesh for {csv_file}. Skipping.")
            continue

        df = pd.read_csv(csv_file)
        
        # Normalize CSA and Volume by their initial values
        csa_norm = df["csa"] / df["csa"].iloc[0]
        volume_norm = df["volume"] / df["volume"].iloc[0]
        
        time_weeks = df["t"] / (24 * 7)

        # Plot normalized CSA
        axes[0].plot(time_weeks, csa_norm, ls=linestyles[i], color=color_map[mesh_name], label=mesh_name)
        
        # Plot normalized Volume
        axes[1].plot(time_weeks, volume_norm, ls=linestyles[i], color=color_map[mesh_name])

    # --- Final Figure Formatting ---
    # Set y labels for each subplot
    axes[0].set_ylabel("Normalized CSA")
    axes[1].set_ylabel("Normalized Volume")

    # Common settings for all subplots
    for ax in axes:
        ax.set_xlabel("Time (weeks)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.minorticks_off()

    # Get handles and labels from the first plot to create a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Place the legend to the right of the subplots
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5), title="Mesh refinement")

    # Adjust subplot layout to make room for the legend
    plt.tight_layout()
    
    # Save the figures (using bbox_inches='tight' prevents legend cutoff)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.svg'), bbox_inches='tight')
    print(f"Mesh convergence figure saved to: {output_file}")


if __name__== "__main__":
    app()
