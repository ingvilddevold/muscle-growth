import typer
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing_extensions import Annotated
import numpy as np
import gc
import dolfinx
import adios4dolfinx
from mpi4py import MPI
from typing import List

# --- Matplotlib Styling ---
import scienceplots
plt.style.use("science")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 7
plt.rcParams["text.usetex"] = False
plt.rcParams['svg.fonttype'] = 'none' # makes text editable in Inkscape
# --- End Styling ---

# Define the main Typer application
app = typer.Typer()


def process_ode_states_sequentially(
    ode_bp_files: list[Path], 
    labels: list[str],
    ode_states: list[str]
) -> pd.DataFrame:
    """
    Loads specified ODE state data from BP files sequentially, calculates aggregates
    (mean, min, max per time step) of the RAW values, and returns aggregated data.
    """
    all_agg_dfs = []
    domain = None # Load mesh once
    V_ode = None  # Function space for ODE states (assuming all are DG0)

    for bp_file, label in zip(ode_bp_files, labels):
        try:
            print(f"  Processing ODE states for: {label} ({bp_file.name})")

            # 1. Read Mesh (if not already done) and get times
            if domain is None:
                domain = adios4dolfinx.read_mesh(bp_file, MPI.COMM_WORLD, engine="BP4", ghost_mode=dolfinx.mesh.GhostMode.none)
                # Assume all ODE states are "DG" 0 (cell-wise constant)
                V_ode = dolfinx.fem.functionspace(domain, ("DG", 0)) 

            # Load time from ode_times.npy
            times = np.load(bp_file.parent / "ode_times.npy")
            
            # 2. Prepare functions and storage for aggregates
            # Create a dictionary of dolfinx Functions, one for each state
            ode_funcs = {state: dolfinx.fem.Function(V_ode, name=state) for state in ode_states}
            
            # Create a dictionary of lists to store aggregated data
            # { "igf1": {'time': [], 'mean': [], ...}, "akt": {...}, ... }
            agg_data = {
                state: {'Time (weeks)': [], 'mean': [], 'min': [], 'max': []} 
                for state in ode_states
            }
            # 3. Iterate through time steps, read data, calculate stats
            print(f"    Reading {len(times)} time steps for {len(ode_states)} states...")
            for i, t in enumerate(times):
                
                for state in ode_states:
                    func = ode_funcs[state]
                    
                    # Read data for the specific function (by name) at time t
                    adios4dolfinx.read_function(bp_file, func, time=t)
                    data_array = func.x.array
                    
                    # Store aggregates of the raw data (no normalization)
                    agg_data[state]['Time (weeks)'].append(t / (24 * 7))
                    agg_data[state]['mean'].append(np.mean(data_array))
                    agg_data[state]['min'].append(np.min(data_array))
                    agg_data[state]['max'].append(np.max(data_array))

            # 4. Create aggregated DataFrame for this file
            file_dfs = []
            for state in ode_states:
                df_state = pd.DataFrame(agg_data[state])
                df_state['state'] = state     # Add column to identify the state
                df_state['variation'] = label # Add column to identify the variation
                file_dfs.append(df_state)
            
            df_agg = pd.concat(file_dfs, ignore_index=True)
            all_agg_dfs.append(df_agg)

            # 5. --- Free Memory ---
            del ode_funcs, agg_data, df_agg, file_dfs
            gc.collect()

        except Exception as e:
            print(f"Warning: Error processing ODE data {bp_file}: {e}. Skipping.")

    if not all_agg_dfs: return pd.DataFrame()
    print("Concatenating aggregated ODE data...")
    final_aggregated_df = pd.concat(all_agg_dfs, ignore_index=True)
    return final_aggregated_df


@app.command("aggregate")
def aggregate_data(
    ode_bp_files: Annotated[str, typer.Option(help="Comma-separated paths to ode_spatial_history.bp files.")],
    variation_labels: Annotated[str, typer.Option(help="Comma-separated string of variation labels.")],
    output_csv: Annotated[Path, typer.Option(help="Path to save the aggregated data CSV file.")],
):
    """
    Aggregates spatial ODE states (IGF1, AKT, FOXO, mTOR, z) from BP files 
    and saves the combined data to a single CSV file.
    """
    bp_files = [Path(p) for p in ode_bp_files.split(',')] # Convert to Path objects
    labels = variation_labels.split(',')

    if not (len(bp_files) == len(labels)):
        print(f"Error: Mismatch in number of files ({len(bp_files)}) and labels ({len(labels)}).")
        raise typer.Exit(code=1)

    # Define the states to aggregate (now including 'z')
    ode_states_to_aggregate = ["igf1", "akt", "foxo", "mtor", "z"]

    # Process ODE data sequentially (memory-efficient)
    print("Processing spatial ODE states sequentially...")
    # ode_agg_df contains columns: 'Time (weeks)', 'mean', 'min', 'max', 'state', 'variation'
    ode_agg_df = process_ode_states_sequentially(bp_files, labels, ode_states_to_aggregate)

    if ode_agg_df.empty:
        print("Error: No valid ODE data loaded. Exiting.")
        raise typer.Exit(code=1)

    # Save the aggregated data
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    ode_agg_df.to_csv(output_csv, index=False)
    print(f"Successfully aggregated data and saved to {output_csv}")

@app.command("plot")
def plot_data(
    input_csv: Annotated[Path, typer.Option(help="Path to the aggregated CSV file.")],
    output_file: Annotated[Path, typer.Option(help="Path to save the output plot.")],
):
    """
    Plots aggregated statistics across variations:
    1. Baseline (Red line)
    2. Variations Ensemble (Grey/Black):
       - Mean of Means +/- Std
       - Mean of Max +/- Std
       - Mean of Min +/- Std
    """
    
    print(f"Loading aggregated data from {input_csv}...")
    try:
        ode_agg_df = pd.read_csv(input_csv)
    except FileNotFoundError:
        print(f"Error: Input file not found.")
        raise typer.Exit(code=1)

    # --- 1. Separate Baseline from Variations ---
    # We assume 'baseline' is the control, and everything else is the ensemble
    baseline_df = ode_agg_df[ode_agg_df['variation'] == 'baseline']
    variations_df = ode_agg_df[ode_agg_df['variation'] != 'baseline']

    if variations_df.empty:
        print("Error: No variation data found to aggregate.")
        raise typer.Exit(code=1)

    # --- 2. Aggregate the Variations (The Ensemble) ---
    # Group by state and time, then calculate mean and std for mean, min, and max columns
    print("Aggregating ensemble statistics...")
    ensemble_stats = variations_df.groupby(['state', 'Time (weeks)']).agg(
        mean_of_means=('mean', 'mean'),
        std_of_means=('mean', 'std'),
        mean_of_max=('max', 'mean'),
        std_of_max=('max', 'std'),
        mean_of_min=('min', 'mean'),
        std_of_min=('min', 'std')
    ).reset_index()

    # --- 3. Plotting Setup ---
    print(f"Generating plot...")
    fig, axes = plt.subplots(2, 3, figsize=(5, 3))
    axes_flat = axes.flatten()

    plot_map = {
        "igf1": axes_flat[0], "akt": axes_flat[1],
        "foxo": axes_flat[3], "mtor": axes_flat[4],
        "z": axes_flat[5]
    }
    plot_titles = {
        "igf1": "IGF1", "akt": "AKT", "foxo": "FOXO",
        "mtor": "mTOR", "z": "Myofibrils N"
    }

    # Colors
    baseline_color = "#1b699c"  # DeFreitas blue
    ensemble_color = "#333333"  # Dark Grey/Black
    
    legend_elements = [] # Custom legend handles

    for i, (state_name, ax) in enumerate(plot_map.items()):
        # Data for this state
        e_data = ensemble_stats[ensemble_stats['state'] == state_name]
        b_data = baseline_df[baseline_df['state'] == state_name]

        if e_data.empty:
            ax.text(0.5, 0.5, "No Data", transform=ax.transAxes, ha='center')
            continue

        t = e_data['Time (weeks)']

        # --- A. Plot Ensemble (The 3 Bands) ---
        
        # 1. Mean of Means (Central Band)
        # Solid line for the average behavior
        ax.plot(t, e_data['mean_of_means'], color=ensemble_color, lw=1.0, zorder=3)
        # Shading for Std of Means
        ax.fill_between(
            t, 
            e_data['mean_of_means'] - e_data['std_of_means'],
            e_data['mean_of_means'] + e_data['std_of_means'],
            color=ensemble_color, alpha=0.4, zorder=2, lw=0
        )

        # 2. Mean of Max (Upper Band)
        # Dashed line for the average Max
        ax.plot(t, e_data['mean_of_max'], color=ensemble_color, ls='--', lw=0.8, alpha=0.7, zorder=2)
        # Shading for Std of Max
        ax.fill_between(
            t, 
            e_data['mean_of_max'] - e_data['std_of_max'],
            e_data['mean_of_max'] + e_data['std_of_max'],
            color=ensemble_color, alpha=0.15, zorder=1, lw=0
        )

        # 3. Mean of Min (Lower Band)
        # Dashed line for the average Min
        ax.plot(t, e_data['mean_of_min'], color=ensemble_color, ls='--', lw=0.8, alpha=0.7, zorder=2)
        # Shading for Std of Min
        ax.fill_between(
            t, 
            e_data['mean_of_min'] - e_data['std_of_min'],
            e_data['mean_of_min'] + e_data['std_of_min'],
            color=ensemble_color, alpha=0.15, zorder=1, lw=0
        )

        # --- B. Plot Baseline (Uniform) ---
        if not b_data.empty:
            ax.plot(
                b_data['Time (weeks)'], b_data['mean'], 
                color=baseline_color, lw=1.0, ls='--', zorder=4,
                label='Uniform'
            )

        # --- C. Styling ---
        ax.set_title(plot_titles.get(state_name, state_name), fontsize=7)
        
        # Thresholds
        #if state_name == "mtor":
        #    ax.axhline(0.434, color='k', ls=':', lw=1)
        #elif state_name == "foxo":
        #    ax.axhline(0.469, color='k', ls=':', lw=1)

        ax.set_ylabel("")
        ax.set_xlabel("Time (weeks)")
        ax.set_xlim(left=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(top=False, right=False)
        ax.minorticks_off()
        # Tweak x-ticks
        ax.set_xticks([0,3,6,9])
        
        # Create custom legend handles only once
        if i == 0:
            from matplotlib.lines import Line2D
            from matplotlib.patches import Patch
            legend_elements = [
                Line2D([0], [0], color=baseline_color, lw=1.0, ls="--", label='Uniform'),
                Line2D([0], [0], color=ensemble_color, lw=1.0, label='Samples Mean'),
                Patch(facecolor=ensemble_color, alpha=0.4, label='Std of Mean'),
                Line2D([0], [0], color=ensemble_color, ls='--', lw=1, alpha=0.8, label='Samples Max/Min'),
                Patch(facecolor=ensemble_color, alpha=0.15, label='Std of Max/Min'),
            ]

    # --- Finalize ---
    axes_flat[2].axis('off') # Hide unused subplot
    
    # Add global legend
    fig.legend(
        handles=legend_elements,
        loc='center left', 
        bbox_to_anchor=(0.75, 0.8),
        #title="Statistics",
        frameon=False
    )
    
    fig.tight_layout()

    # Save the figure
    output_file.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.savefig(output_file.with_suffix('.svg'), bbox_inches='tight') # Also save SVG

    print(f"Successfully saved plot to {output_file} (and .svg)")

if __name__ == "__main__":
    app()
