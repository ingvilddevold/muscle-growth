import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import typer
from typing_extensions import Annotated
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib import rc

# --- Matplotlib style ---
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["svg.fonttype"] = "none"
rc("text", usetex=False)

app = typer.Typer()


def get_muscle_type(mesh_name):
    if "BicepsFemoris" in mesh_name:
        return "BFLH"
    if "Semitendinosus" in mesh_name or "Semitendonosus" in mesh_name: # account for typo
        return "ST"
    if "TibialisAnterior" in mesh_name:
        return "TA"
    return "Unknown"


def get_subject_group(mesh_name):
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
    try:
        df = pd.read_csv(input_file)
    except FileNotFoundError:
        print(f"Error: Input file not found at {input_file}")
        output_file.touch()
        return

    # --- 2. Prepare Data ---
    df["muscle_type"] = df["mesh"].apply(get_muscle_type)
    df["subject_group"] = df["mesh"].apply(get_subject_group)
    df["main_subject_group"] = df["mesh"].apply(get_main_subject_group)
    df["gender"] = df["subject_group"].apply(
        lambda x: "Female" if "Female" in x else "Male"
    )

    df_melted = df.melt(
        id_vars=[
            "mesh",
            "muscle_type",
            "subject_group",
            "main_subject_group",
            "gender",
        ],
        value_vars=["volume_cm3", "csa_cm2"],
        var_name="quantity",
        value_name="value",
    )

    # --- 3. Define Palettes ---

    # Muscle Color Palette
    muscle_palette = {
        "TA": "#B22020",  
        "BFLH": "#FF2752", 
        "ST": "#FF7429",  
    }

    # X-Axis Label Map
    x_axis_labels = {
        "TA": "TA",
        "BFLH": "BFLH",
        "ST": "ST"
    }

    # Markers for scatter
    gender_markers = {"Female": "o", "Male": "^"}
    subject_group_palette = {
        "Female Left": "#000000",
        "Female Right": "#9ca3af",
        "Male Left": "#000000",
        "Male Right": "#9ca3af",
    }
    subject_group_order = ["Female Left", "Female Right", "Male Left", "Male Right"]

    # --- 4. Create the Plot ---
    plt.style.use("seaborn-v0_8-white")

    muscle_type_order = sorted(df["muscle_type"].unique())

    g = sns.FacetGrid(
        data=df_melted,
        row="quantity",
        row_order=["volume_cm3", "csa_cm2"],
        height=1.7,
        aspect=1.15,
        sharey=False,
        sharex=False,
    )

    # Iterate over axes to plot
    for ax, col_name in zip(g.axes.flat, ["volume_cm3", "csa_cm2"]):
        data_subset = df_melted[df_melted["quantity"] == col_name]

        # A. Draw Base Bar Plots
        sns.barplot(
            data=data_subset,
            x="muscle_type",
            y="value",
            hue="main_subject_group",
            hue_order=["VHP Female", "VHP Male"],
            order=muscle_type_order,
            errorbar=None,
            width=0.8,
            alpha=1.0,
            edgecolor="white",
            linewidth=1.5,
            ax=ax,
            dodge=True,
            palette={
                "VHP Female": "gray",
                "VHP Male": "gray",
            },
            legend=False,
        )

        # --- COLOR & HATCH OVERRIDE ---
        hue_groups = ["VHP Female", "VHP Male"]

        for container, gender_label in zip(ax.containers, hue_groups):
            for bar, muscle_name in zip(container, muscle_type_order):
                # 1. Set Color based on Muscle Type
                bar.set_facecolor(muscle_palette[muscle_name])

                # 2. Apply Hatch if Male
                if "Male" in gender_label:
                    bar.set_hatch("////")
                    bar.set_edgecolor("white")

        # B. Draw Scatter Plots
        sns.stripplot(
            data=data_subset[data_subset["gender"] == "Female"],
            x="muscle_type",
            y="value",
            hue="subject_group",
            order=muscle_type_order,
            hue_order=subject_group_order,
            marker=gender_markers["Female"],
            palette=subject_group_palette,
            linewidth=0.3,
            edgecolor="white",
            dodge=True,
            jitter=0.15,
            s=5,
            alpha=0.8,
            ax=ax,
            legend=False,
        )

        sns.stripplot(
            data=data_subset[data_subset["gender"] == "Male"],
            x="muscle_type",
            y="value",
            hue="subject_group",
            order=muscle_type_order,
            hue_order=subject_group_order,
            marker=gender_markers["Male"],
            palette=subject_group_palette,
            linewidth=0.3,
            edgecolor="white",
            dodge=True,
            jitter=0.15,
            s=5,
            alpha=0.8,
            ax=ax,
            legend=False,
        )

        ax.grid(axis="y", linestyle="--", alpha=0.7)

    # --- 5. Improve Plot Aesthetics ---
    g.set_titles("") 

    label_mapping = {
        "volume_cm3": {
            "title": "Muscle Volume", 
            "ylabel": r"Volume (cm$^3$)"
        },
        "csa_cm2": {
            "title": "Cross-Sectional Area", 
            "ylabel": r"CSA (cm$^2$)"
        }
    }

    # Apply titles and y-labels
    for ax, row_name in zip(g.axes.flat, g.row_names):
        if row_name in label_mapping:
            config = label_mapping[row_name]
            ax.set_title(config["title"], fontsize=8, pad=10)
            ax.set_ylabel(config["ylabel"], fontsize=8)

    # --- Apply X-Axis labels ---
    # Create a list of labels in the correct order corresponding to 'muscle_type_order'
    new_labels = [x_axis_labels.get(m, m) for m in muscle_type_order]
    
    # Apply to the FacetGrid (this applies to the bottom-most axis by default)
    g.set_xticklabels(new_labels, fontsize=8, rotation=0)

    g.set(xlabel=None) # remove Muscle Type label
    g.set_yticklabels(fontsize=8)
    g.despine()

    # --- 6. Custom legend ---
    legend_elements = [

        # --- Section 1: Gender Mean (Pattern) ---
        Patch(facecolor="gray", label="Female Mean"),
        Patch(
            facecolor="gray",
            hatch="////",
            edgecolor="white",
            label="Male Mean",
        ),
        Line2D([0], [0], marker=None, color="none", label=""),

        # --- Section 2: Individual Data ---
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Female Left",
            markerfacecolor=subject_group_palette["Female Left"],
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Female Right",
            markerfacecolor=subject_group_palette["Female Right"],
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Male Left",
            markerfacecolor=subject_group_palette["Male Left"],
            markersize=5,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Male Right",
            markerfacecolor=subject_group_palette["Male Right"],
            markersize=5,
        ),
    ]

    leg = g.fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        frameon=False,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    g.savefig(output_file, dpi=300)
    g.savefig(output_file.with_suffix(".svg"))
    print(f"Plot saved to {output_file}")
    plt.close(g.fig)


if __name__ == "__main__":
    app()
