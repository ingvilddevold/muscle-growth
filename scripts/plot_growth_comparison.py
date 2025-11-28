import typer
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from typing import List

# Matplotlib styling
from matplotlib import pyplot as plt
import scienceplots
plt.style.use("science")
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 8
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
    protocols: str = typer.Option(
        ...,
        "--protocols",
        help="Comma-separated list of protocol names, in the same order as csv-files.",
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
    Generates a 3-panel figure comparing CSA, Volume, and kM over time
    for different growth protocols.
    """
    protocol_names = protocols.split(',')

    # --- Plotting Setup ---
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))
    colors = {"defreitas": "C0", "weekly": "C2", "everythreedays": "C1", "testing": "C3"}
    labels = {"defreitas": "MWF", "weekly": "Weekly", "everythreedays": "Every three days", "testing": "Testing"}

    # --- Process and Plot Data ---
    for csv_file in csv_files:
        protocol_name = next((p for p in protocol_names if p in str(csv_file)), None)
        if not protocol_name:
            print(f"Warning: Could not determine protocol for {csv_file}. Skipping.")
            continue

        df = pd.read_csv(csv_file)
        
        # Normalize CSA and Volume by their initial values
        csa_norm = df["csa"] / df["csa"].iloc[0]
        volume_norm = df["volume"] / df["volume"].iloc[0]
        
        time_weeks = df["t"] / (24 * 7)

        # Plot normalized CSA
        axes[0].plot(time_weeks, csa_norm, color=colors[protocol_name], label=labels[protocol_name])
        
        # Plot normalized Volume
        axes[1].plot(time_weeks, volume_norm, color=colors[protocol_name])

        # Plot kM (k1)
        axes[2].plot(time_weeks, df["k1"], color=colors[protocol_name])

    # --- Final Figure Formatting ---
    # Set y labels for each subplot
    axes[0].set_ylabel("Normalized CSA")
    axes[1].set_ylabel("Normalized Volume")
    axes[2].set_ylabel(r"$k_M$")

    # Common settings for all subplots
    for ax in axes:
        ax.set_xlabel("Time (weeks)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.minorticks_off()
        ax.set_xticks([0, 3, 6, 9])

    # Get handles and labels from the first plot to create a single legend
    handles, labels = axes[0].get_legend_handles_labels()
    
    # Place the legend to the right of the subplots
    fig.legend(handles, labels, loc='center left', bbox_to_anchor=(1.0, 0.5))

    # Adjust subplot layout to make room for the legend on the right
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.savefig(output_file.with_suffix('.svg'))
    print(f"Growth comparison figure saved to: {output_file}")


if __name__== "__main__":
    app()
