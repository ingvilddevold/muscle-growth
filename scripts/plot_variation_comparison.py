import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing_extensions import Annotated
import numpy as np

# --- Matplotlib Styling ---
import scienceplots

plt.style.use("science")
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 7
plt.rcParams["svg.fonttype"] = "none"  # make text editable in Inkscape
rc("text", usetex=False)
# --- End Styling ---


def load_growth_data(csv_files: list[str], labels: list[str]) -> pd.DataFrame:
    """Loads and concatenates the 'growth_results.csv' files."""
    all_dfs = []
    for csv_file, label in zip(csv_files, labels):
        try:
            df = pd.read_csv(csv_file)
            df["variation"] = label
            # Normalize to initial value
            df["csa_norm"] = df["csa"] / df["csa"].iloc[0]
            df["volume_norm"] = df["volume"] / df["volume"].iloc[0]
            df["k1"] = df["k1"]
            df["Time (weeks)"] = df["t"] / (24 * 7)
            all_dfs.append(df)
        except Exception as e:
            print(f"Warning: Error loading {csv_file}: {e}. Skipping.")
    if not all_dfs:
        return pd.DataFrame()
    return pd.concat(all_dfs, ignore_index=True)


@typer.run
def main(
    input_csvs: Annotated[
        str, typer.Option(help="Comma-separated string of 'growth_results.csv' paths.")
    ],
    variation_labels: Annotated[
        str, typer.Option(help="Comma-separated string of variation labels.")
    ],
    output_file: Annotated[Path, typer.Option(help="Path to save the output plot.")],
    protocol: Annotated[
        str, typer.Option(help="Protocol name (optional, not currently used in title).")
    ] = "Protocol",
):
    """
    Loads growth CSVs and plots:
    1. Baseline (Dashed blue line)
    2. Ensemble of Variations (Mean +/- Std, and Range)
    """
    csv_files = input_csvs.split(",")
    labels = variation_labels.split(",")

    if len(csv_files) != len(labels):
        print(
            f"Error: Mismatch in number of files ({len(csv_files)}) and labels ({len(labels)})."
        )
        raise typer.Exit(code=1)

    # --- 1. Load Data ---
    print("Loading growth results...")
    growth_df = load_growth_data(csv_files, labels)

    if growth_df.empty:
        print("Error: No valid data loaded.")
        raise typer.Exit(code=1)

    # --- 2. Separate Baseline from Ensemble ---
    baseline_df = growth_df[growth_df["variation"] == "baseline"]
    variations_df = growth_df[growth_df["variation"] != "baseline"]

    if variations_df.empty:
        print("Error: No variation data found to calculate ensemble statistics.")
        raise typer.Exit(code=1)

    # --- 3. Calculate Ensemble Statistics ---
    print("Calculating ensemble statistics...")

    # We define a dictionary to map the columns we want to aggregate
    # to the new prefix names in the stats dataframe
    agg_dict = {
        "csa_norm": ["mean", "std", "min", "max"],
        "volume_norm": ["mean", "std", "min", "max"],
        "k1": ["mean", "std", "min", "max"],
    }

    # Perform aggregation
    ensemble_stats = variations_df.groupby("Time (weeks)").agg(agg_dict)

    # Flatten MultiIndex columns (e.g., ('csa_norm', 'mean') -> 'csa_norm_mean')
    ensemble_stats.columns = [
        "_".join(col).strip() for col in ensemble_stats.columns.values
    ]
    ensemble_stats = ensemble_stats.reset_index()

    # --- 4. Plotting ---
    print(f"Generating plot...")
    fig, axes = plt.subplots(1, 3, figsize=(5, 1.5))

    # Define plot settings
    baseline_color = "#1b699c"  # DeFreitas blue
    ensemble_color = "#333333"  # Dark Grey/Black

    # Configuration list: (Axis, Stat_Column_Prefix, DataFrame_Column, Y_Label)
    plot_configs = [
        (axes[0], "csa_norm", "csa_norm", "Normalized CSA"),
        (axes[1], "volume_norm", "volume_norm", "Normalized Volume"),
        (axes[2], "k1", "k1", "$k_M$"),
    ]

    legend_elements = []

    for i, (ax, prefix, df_col, ylabel) in enumerate(plot_configs):
        t = ensemble_stats["Time (weeks)"]

        # Construct column names for stats
        mean_col = f"{prefix}_mean"
        std_col = f"{prefix}_std"
        min_col = f"{prefix}_min"
        max_col = f"{prefix}_max"

        # --- Plot Ensemble ---
        if mean_col in ensemble_stats.columns:
            # 1. Mean Line
            ax.plot(t, ensemble_stats[mean_col], color=ensemble_color, lw=1.5, zorder=3)

            # 2. Std Dev Shading
            ax.fill_between(
                t,
                ensemble_stats[mean_col] - ensemble_stats[std_col],
                ensemble_stats[mean_col] + ensemble_stats[std_col],
                color=ensemble_color,
                alpha=0.3,
                zorder=2,
                lw=0,
            )

            # 3. Min/Max Range (Dashed Lines)
            ax.plot(
                t,
                ensemble_stats[min_col],
                color=ensemble_color,
                ls="--",
                lw=0.8,
                alpha=0.5,
                zorder=2,
            )
            ax.plot(
                t,
                ensemble_stats[max_col],
                color=ensemble_color,
                ls="--",
                lw=0.8,
                alpha=0.5,
                zorder=2,
            )

        # --- Plot Baseline ---
        if not baseline_df.empty and df_col in baseline_df.columns:
            ax.plot(
                baseline_df["Time (weeks)"],
                baseline_df[df_col],
                color=baseline_color,
                lw=1.5,
                ls="--",
                zorder=4,
            )

        # --- Styling ---
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Time (weeks)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.minorticks_off()
        ax.set_xticks([0, 3, 6, 9])
        ax.set_xlim(left=0)

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.4)

    # Save
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.savefig(output_file.with_suffix(".svg"), bbox_inches="tight")

    print(f"Successfully saved plot to {output_file} (and .svg)")
