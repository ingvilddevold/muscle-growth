import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.markers import MarkerStyle


def plot_sa_scatter(ax, sa_results_csv):
    """
    Generates a scatter plot of sensitivity analysis results on a given
    matplotlib axis.

    Args:
        ax: The matplotlib axis object to plot on.
        sa_results_csv: Path to the CSV file containing ST and ST_conf values.
    """
    # --- Data Loading and preparation ---
    sensitivity_results = pd.read_csv(sa_results_csv, index_col=0)
    ST = sensitivity_results["ST"]
    ST_conf = sensitivity_results["ST_conf"]

    # Convert confidence intervals to standard deviation through Z-score
    from scipy.stats import norm
    conf_level = 0.95
    Z = norm.ppf(0.5 + conf_level / 2)
    ST_std = ST_conf / Z


    COLORS = {
        "igf1": "#D8CC39",
        "akt": "#46A0D4",
        "foxo": "#BF4A00",
        "mtor": "#00885F",
    }

    parameter_names_latex = [
        r"$a_{I}^{0}$",
        r"$a_{A}^{0}$",
        r"$a_{F}$",
        r"$a_{M}$",
        r"$b_{I}$",
        r"$b_{A}$",
        r"$b_{F}$",
        r"$b_{M}$",
        r"$c_{AI}$",
        r"$c_{FA}$",
        r"$c_{MA}$",
        r"$c_{MF}$",
        r"$k_{M}$",
        r"$k_{F}$",
    ]
    parameter_names = [
        "a_I0",
        "a_A0",
        "a_F",
        "a_M",
        "b_I",
        "b_A",
        "b_F",
        "b_M",
        "c_AI",
        "c_FA",
        "c_MA",
        "c_MF",
        "k_M",
        "k_F",
    ]

    # Exclude k_M and k_F from the plot
    filtered_parameter_names = [p for p in parameter_names if p not in ["k_M", "k_F"]]
    filtered_indices = [
        ST[i] for i, p in enumerate(parameter_names) if p not in ["k_M", "k_F"]
    ]
    filtered_confidences = [
        ST_conf[i] for i, p in enumerate(parameter_names) if p not in ["k_M", "k_F"]
    ]
    filtered_sd = [
        ST_std[i] for i, p in enumerate(parameter_names) if p not in ["k_M", "k_F"]
    ]

    # Define which parameters belong to which groups
    parameter_groups = {
        "IGF1": ["a_I0", "b_I", "c_AI"],
        "AKT": ["a_A0", "b_A", "c_FA", "c_MA", "c_AI"],
        "FOXO": ["a_F", "b_F", "c_MF", "c_FA"],
        "mTOR": ["a_M", "b_M", "c_MF", "c_MA"],
    }
    group_colors = {
        "IGF1": COLORS["igf1"],
        "AKT": COLORS["akt"],
        "FOXO": COLORS["foxo"],
        "mTOR": COLORS["mtor"],
    }
    param_to_groups = {p: set() for p in filtered_parameter_names}
    for group, params in parameter_groups.items():
        for param in params:
            if param in param_to_groups:
                param_to_groups[param].add(group)

    filtered_parameter_names_latex = [
        parameter_names_latex[parameter_names.index(name)]
        for name in filtered_parameter_names
    ]

    # Only annotate the top 6 parameters based on their indices
    top6_indices = np.argsort(filtered_indices)[-6:]
    offsets = [(-10, 8), (0, 5), (-5, 7), (0, 5), (0, 5), (0, 5)]

    # --- Plotting ---
    for i, name in enumerate(filtered_parameter_names):
        groups = list(param_to_groups.get(name, set()))
        #x, y = filtered_indices[i], filtered_confidences[i]
        x, y = filtered_indices[i], filtered_sd[i]
        s = 30  # marker size
        if len(groups) == 2:
            # Plot two-colored markers for shared parameters (coupling strengths)
            ax.scatter(
                x,
                y,
                s=s,
                c=group_colors[groups[0]],
                marker=MarkerStyle("o", fillstyle="left"),
                zorder=4,
                edgecolors="black",
                linewidths=0.7,
            )
            ax.scatter(
                x,
                y,
                s=s,
                c=group_colors[groups[1]],
                marker=MarkerStyle("o", fillstyle="right"),
                zorder=4,
                edgecolors="black",
                linewidths=0.7,
            )
        elif len(groups) == 1:
            # Plot single-colored markers for parameters belonging to one group
            # (intrinsic growth rates and self-inhibition rates)
            ax.scatter(
                x,
                y,
                s=s,
                c=group_colors[groups[0]],
                marker="o",
                zorder=4,
                edgecolors="black",
                linewidths=0.7,
            )
        else:
            ax.scatter(
                x,
                y,
                s=s,
                c="gray",
                marker="o",
                zorder=4,
                edgecolors="black",
                linewidths=0.7,
            )

        if i in top6_indices:
            ax.annotate(
                filtered_parameter_names_latex[i],
                (x, y),
                textcoords="offset points",
                xytext=offsets[i % len(offsets)],
                fontsize=8,
                usetex=True,
            )

    # --- Legend and styling ---
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label=group,
            markerfacecolor=color,
            markersize=(s**0.5),  # matching size of scatter markers
            markeredgecolor="black",
            markeredgewidth=0.7,
        )
        for group, color in group_colors.items()
    ]
    ax.legend(
        handles=legend_elements,
        title="",
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1.05, 0),
        fontsize=8,
        labelspacing=0.3,
        handletextpad=0.05,
    )
    ax.set_xlabel(r"$\mu$ (Mean Total-Order Index)")
    ax.set_ylabel(r"$\sigma$ (Std. Dev.)")

    # increase ylim slightly to make room for the top markers
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(y_min, y_max * 1.1)
