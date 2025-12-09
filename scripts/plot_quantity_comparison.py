import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import typer
from typing_extensions import Annotated
from typing import List
import json
from matplotlib.lines import Line2D  # Import for custom legend

# Matplotlib configuration
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 7
plt.rcParams["svg.fonttype"] = "none"
rc("text", usetex=False)

# Create a Typer application
app = typer.Typer()


def get_protocol_and_geometry(path, protocols, geometries):
    """Extract protocol and geometry from a given path."""
    path_str = str(path)
    for p in protocols:
        if p in path_str:
            protocol = p
            break
    else:
        return None, None

    for g in geometries:
        if g in path_str:
            geometry = g
            break
    else:
        return None, None
    return protocol, geometry


def get_subject_group(geometry_name):
    """Determine the subject group from the geometry name."""
    if "VHF" in geometry_name:
        group = "Female"
    elif "VHM" in geometry_name:
        group = "Male"
    else:
        # Default case if no specific group is identified
        return "Unknown"

    if "Left" in geometry_name:
        group += " Left"
    elif "Right" in geometry_name:
        group += " Right"

    return group


@app.command()
def main(
    input_files: Annotated[
        List[Path], typer.Argument(help="Paths to the input CSV files.")
    ],
    output_file: Annotated[Path, typer.Option(help="Path to the output plot file.")],
    muscle_groups_json: Annotated[
        str, typer.Option(help="Muscle groups as a JSON string.")
    ],
    protocols_str: Annotated[
        str, typer.Option(help="Comma-separated list of protocols.")
    ],
    geometries_str: Annotated[
        str, typer.Option(help="Comma-separated list of geometries.")
    ],
    quantity_to_plot: Annotated[
        str, typer.Option(help="The quantity to plot from the CSV files.")
    ],
):
    """
    Generates a comparative plot of a normalized quantity for different muscle geometries,
    creating separate panels for each protocol.
    """
    # Parse command-line arguments that are passed as strings
    muscle_groups = json.loads(muscle_groups_json)
    protocols = protocols_str.split(",")
    geometries = geometries_str.split(",")

    quantity_to_plot_name = quantity_to_plot
    quantity_to_plot = quantity_to_plot.lower()

    data = []
    for file_path in input_files:
        protocol, geometry = get_protocol_and_geometry(file_path, protocols, geometries)
        if not protocol or not geometry:
            print(f"Warning: Could not determine protocol/geometry for {file_path}")
            continue
        muscle_type = muscle_groups.get(geometry)
        if not muscle_type:
            print(
                f"Warning: Geometry '{geometry}' not found in muscle groups. Skipping."
            )
            continue
        subject_group = get_subject_group(geometry)
        try:
            df = pd.read_csv(file_path)
            if (
                quantity_to_plot in df.columns
                and not df.empty
                and len(df[quantity_to_plot]) > 0
            ):
                initial_val = df[quantity_to_plot].iloc[0]
                final_val = df[quantity_to_plot].iloc[-1]

                if initial_val > 0:
                    normalized_val = final_val / initial_val
                    data.append(
                        {
                            "protocol": protocol,
                            "geometry": geometry,
                            "muscle_type": muscle_type,
                            f"normalized_{quantity_to_plot}": normalized_val,
                            "subject_group": subject_group,
                        }
                    )
                else:
                    print(
                        f"Warning: Initial {quantity_to_plot} is zero in {file_path}, cannot normalize."
                    )
            else:
                print(
                    f"Warning: '{quantity_to_plot}' column not found or empty in {file_path}"
                )
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

    df_plot = pd.DataFrame(data)

    # --- Split subject_group for styling ---
    df_plot["gender"] = df_plot["subject_group"].apply(
        lambda x: "Female" if "Female" in x else "Male"
    )
    df_plot["side"] = df_plot["subject_group"].apply(
        lambda x: "Left" if "Left" in x else "Right"
    )

    # --- Define palettes and markers for plot ---
    gender_markers = {"Female": "o", "Male": "^"}  # Circle  # Triangle

    subject_group_order = ["Female Left", "Female Right", "Male Left", "Male Right"]

    # Colors for markers
    subject_group_palette = {
        "Female Left": "#000000",  # Strong Pink-Red
        "Female Right": "#9ca3af",  # Light Pink
        "Male Left": "#000000",  # Strong Pink-Red (same as female left)
        "Male Right": "#9ca3af",  # Light Pink (same as female right)
    }

    # Colors for bars
    protocol_palettes = {
        "defreitas": [
            "#1b699c",
            "#1f77b4",
            "#4c91c3",
        ],  # C0: Dark Blue, C0 Base, Light Blue
        "everythreedays": [
            "#278f27",
            "#2ca02c",
            "#56b356",
        ],  # C2: Dark Green, C2 Base, Light Green
        "weekly": [
            "#e6730d",
            "#ff7f0e",
            "#ff953d",
        ],  # C1: Dark Orange, C1 Base, Light Orange
    }

    # --- Create the plot ---
    plt.style.use("seaborn-v0_8-white")

    # Define the order of protocols and muscle types for consistent plotting
    protocol_order = sorted(df_plot["protocol"].unique())
    muscle_type_order = sorted(df_plot["muscle_type"].unique())

    # Create a FacetGrid to manage the subplots for each protocol
    g = sns.FacetGrid(
        data=df_plot,
        col="protocol",
        col_order=protocol_order,
        height=2.0,
        aspect=0.7,
        sharey=False,
    )

    for ax, protocol in zip(g.axes.flat, protocol_order):
        # Get the data for this specific subplot
        data_subset = df_plot[df_plot["protocol"] == protocol]
        current_palette = protocol_palettes.get(protocol)

        # --- 1. Draw the bar plot ---
        sns.barplot(
            data=data_subset,
            x="muscle_type",
            y=f"normalized_{quantity_to_plot}",
            order=muscle_type_order,
            palette=current_palette,
            alpha=1.0,
            errorbar="se",
            capsize=0.1,
            err_kws={"linewidth": 1.5},
            ax=ax,
        )

        # --- Set Y-limits individually for this subplot ---
        # y_values = data_subset[f"normalized_{quantity_to_plot}"]
        # max_val = y_values.max()
        # min_val = y_values.min()
        #
        ## Calculate a 10% margin above the max value (relative to 1.0)
        # margin = (max_val - 1.0) * 0.1
        # if margin < 0.005: # Ensure a minimum margin
        #    margin = 0.005
        #
        ## Set the local y-limit with a fixed bottom
        # ax.set_ylim(min_val - margin, max_val + margin)

        # --- 2. Draw the scatter plots ---

        # Plot females (circles)
        sns.stripplot(
            data=data_subset[data_subset["gender"] == "Female"],  # Filter for Females
            x="muscle_type",
            y=f"normalized_{quantity_to_plot}",
            hue="subject_group",
            order=muscle_type_order,
            hue_order=subject_group_order,
            marker=gender_markers["Female"],
            palette=subject_group_palette,
            linewidth=0.3,
            dodge=True,
            jitter=0.2,
            s=5,
            alpha=0.8,
            ax=ax,  # Plot on this axis
            legend=False,  # Disable individual plot legends
        )

        # Plot males (triangles)
        sns.stripplot(
            data=data_subset[data_subset["gender"] == "Male"],  # Filter for Males
            x="muscle_type",
            y=f"normalized_{quantity_to_plot}",
            hue="subject_group",
            order=muscle_type_order,
            hue_order=subject_group_order,
            marker=gender_markers["Male"],
            palette=subject_group_palette,
            linewidth=0.3,
            dodge=True,
            jitter=0.2,
            s=5,
            alpha=0.8,
            ax=ax,  # Plot on this axis
            legend=False,  # Disable individual plot legends
        )

    # --- Improve Plot Aesthetics ---
    min_val = df_plot[f"normalized_{quantity_to_plot}"].min()
    max_val = df_plot[f"normalized_{quantity_to_plot}"].max()
    margin = (max_val - min_val) * 0.05

    # Set y-axis limits for all subplots
    g.set(ylim=(min_val - margin, max_val + margin))

    quantity_label = quantity_to_plot_name.replace("_", " ")

    # Add a main title for the entire figure
    g.fig.suptitle(
        f"Final Normalized {quantity_label} by Protocol and Muscle Type",
        y=0.9,
        fontsize=8,
    )

    # Set axis labels for the entire figure
    g.set_axis_labels(x_var="", y_var=f"Normalized {quantity_label}")

    # Set titles for each subplot
    protocol_labels = {
        "defreitas": "MWF",
        "weekly": "Weekly",
        "everythreedays": "Every three days",
        "testing": "Testing",
    }
    for ax, title in zip(g.axes.flat, g.col_names):
        ax.set_title(protocol_labels.get(title, title), size=7)

    # g.set_yticklabels(fontsize=8)

    # Rotate x-axis labels
    g.set_xticklabels(rotation=45, ha="right")

    g.despine()

    # ---  Create a custom legend ---
    legend_elements = [
        # Title for "Female"
        Line2D([0], [0], marker=None, color="none", label="Female", linestyle="None"),
        # Female markers
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Left",
            markerfacecolor=subject_group_palette["Female Left"],
            markersize=5,
            linestyle="None",
            markeredgewidth=0.3,
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="Right",
            markerfacecolor=subject_group_palette["Female Right"],
            markersize=5,
            linestyle="None",
            markeredgewidth=0.3,
        ),
        # Spacer
        Line2D([0], [0], marker=None, color="none", label="", linestyle="None"),
        # Title for "Male"
        Line2D([0], [0], marker=None, color="none", label="Male", linestyle="None"),
        # Male markers
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Left",
            markerfacecolor=subject_group_palette["Male Left"],
            markersize=5,
            linestyle="None",
            markeredgewidth=0.3,
        ),
        Line2D(
            [0],
            [0],
            marker="^",
            color="w",
            label="Right",
            markerfacecolor=subject_group_palette["Male Right"],
            markersize=5,
            linestyle="None",
            markeredgewidth=0.3,
        ),
    ]

    # Add the legend to the figure
    leg = g.fig.legend(
        handles=legend_elements,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),  # Positioned to the right
        frameon=False,
    )

    # Make the "Female" and "Male" labels bold
    leg.get_texts()[0].set_fontweight("bold")
    leg.get_texts()[4].set_fontweight("bold")

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.97])

    # Save the figure
    g.savefig(output_file, dpi=300)
    g.savefig(output_file.with_suffix(".svg"))  # Also save SVG
    print(f"Plot saved to {output_file}")


if __name__ == "__main__":
    app()
