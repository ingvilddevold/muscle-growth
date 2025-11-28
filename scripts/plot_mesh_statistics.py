import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import json
import typer
from typing_extensions import Annotated

# Matplotlib configuration
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["CMU Sans Serif"]
rc("text", usetex=False)

# Create a Typer application
app = typer.Typer()


def get_muscle_type(mesh_name):
    """Extract the muscle type (BFLH, ST, TA) from the mesh name."""
    if "BicepsFemoris" in mesh_name:
        return "BFLH"
    if "Semitendinosus" in mesh_name or "Semitendonosus" in mesh_name:
        return "ST"
    if "TibialisAnterior" in mesh_name:
        return "TA"
    return "Unknown"


def get_subject_group(mesh_name):
    """Determine the subject group from the geometry name."""
    if "VHF" in mesh_name:
        group = "Female"
    elif "VHM" in mesh_name:
        group = "Male"
    else:
        return "Unknown"

    if "Left" in mesh_name:
        group += " Left"
    elif "Right" in mesh_name:
        group += " Right"

    return group


def get_main_subject_group(mesh_name):
    """Determine the main subject group (Female/Male) from the geometry name."""
    if "VHF" in mesh_name:
        return "VHP Female"
    elif "VHM" in mesh_name:
        return "VHP Male"
    return "Unknown"


@app.command()
def main(
    input_file: Annotated[
        Path, typer.Option(help="Path to the input mesh_statistics.csv file.")
    ],
    output_file: Annotated[Path, typer.Option(help="Path to the output plot file.")],
):
    """
    Generates bar plots of muscle volume and CSA from a mesh statistics CSV file,
    with individual data points overlaid as a scatter plot.
    """
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        # Create an empty file to satisfy Snakemake if needed
        output_file.touch()
        return

    # --- 1. Prepare the DataFrame for Plotting ---
    df["muscle_type"] = df["mesh"].apply(get_muscle_type)
    df["subject_group"] = df["mesh"].apply(get_subject_group)
    df["main_subject_group"] = df["mesh"].apply(get_main_subject_group)

    df_melted = df.melt(
        id_vars=["mesh", "muscle_type", "subject_group", "main_subject_group"],
        value_vars=["volume_cm3", "csa_cm2"],
        var_name="quantity",
        value_name="value",
    )

    # --- 2. Create the Plot ---
    plt.style.use("seaborn-v0_8-poster")

    # Use FacetGrid for more control over layering plots
    g = sns.FacetGrid(
        data=df_melted,
        col="quantity",
        height=8,
        aspect=0.8,
        sharey=False,  # Use independent y-axes for volume and csa
    )

    subject_groups_palette = {
        "Female Left": "red",
        "Female Right": "lightcoral",
        "Male Left": "blue",
        "Male Right": "lightblue",
    }
    # Derive a simpler palette for the bar plots
    bar_palette = {
        "VHP Female": subject_groups_palette.get("Female Right", "lightcoral"),
        "VHP Male": subject_groups_palette.get("Male Right", "lightblue"),
    }

    # 1. Draw the bar plots for the mean and standard error, grouped by main subject
    g.map_dataframe(
        sns.barplot,
        x="muscle_type",
        y="value",
        hue="main_subject_group",
        palette=bar_palette,
        errorbar=None,
        dodge=True,
        width=0.6,
    )

    # 2. Overlay the scatter plots to show individual data points, colored by specific group
    g.map_dataframe(
        sns.stripplot,
        x="muscle_type",
        y="value",
        hue="subject_group",
        palette=subject_groups_palette,
        edgecolor="black",
        linewidth=0.7,
        dodge=True,
        s=12,
        # jitter=0.1
    )

    # --- 3. Improve Plot Aesthetics ---
    g.fig.suptitle(
        "Muscle Geometric Properties", y=1.03, fontsize=22
    )
    g.set_axis_labels("Muscle Type", "", fontsize=16)

    g.axes.flat[0].set_ylabel(r"Volume (cm$^3$)", fontsize=16)
    g.axes.flat[0].set_title("Volume", size=18)
    g.axes.flat[1].set_ylabel(r"Cross-Sectional Area (cm$^2$)", fontsize=16)
    g.axes.flat[1].set_title("Cross-Sectional Area", size=18)

    g.set_xticklabels(rotation=45, ha="right")
    for ax in g.axes.flat:
        ax.grid(axis="y", linestyle="--", alpha=0.7)

    g.add_legend(title="Subject Group")
    sns.move_legend(g, "center left", bbox_to_anchor=(1.02, 0.5))

    plt.tight_layout(rect=[0, 0, 1, 0.97])

    g.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close(g.fig)

    # --- 4. Create the CSA vs. Volume Scatter Plot ---
    plt.figure(figsize=(8, 6))
    ax_scatter = sns.scatterplot(
        data=df,
        x="csa_cm2",
        y="volume_cm3",
        hue="subject_group",
        palette=subject_groups_palette,
        s=150,
        edgecolor="black",
        linewidth=0.7,
        alpha=0.9,
    )

    output_file2 = output_file.parent / f"{output_file.stem}_csa_vs_volume.png"
    ax_scatter.set_xlabel(r"Cross-Sectional Area (cm$^2$)", fontsize=16)
    ax_scatter.set_ylabel(r"Volume (cm$^3$)", fontsize=16)
    ax_scatter.set_title("Volume vs. Cross-Sectional Area", fontsize=20)

    plt.tight_layout()
    plt.savefig(output_file2, dpi=300)
    print(f"Scatter plot saved to {output_file2}")


if __name__ == "__main__":
    app()
