"""
Script to run parameter fitting for kM (k1) using a cylinder geometry and
the DeFreitas (MWF) protocol.
"""

import musclex
import dolfinx
from musclex.exercise_model import ExerciseModel
from musclex.protocol import DeFreitasProtocol
from musclex.muscle_growth import MuscleGrowthModel
from musclex.material import MuscleRohrle
from pathlib import Path
import pandas as pd
from matplotlib import pyplot as plt
import scienceplots

# --- Setup ---

# Base output directory
base_output_dir = Path(__file__).parents[1] / "results" / "kM_parameter_fit"

# Set up the protocol and geometry
protocol = DeFreitasProtocol()
geometry = musclex.geometry.CylinderGmsh()
# Material config file
config_file = Path(__file__).parents[1] / "config_files/material_rohrle.yml"
# Exercise config file
exercise_config = Path(__file__).parents[1] / "config_files/exercise_eq_reduced_k1.yml"

# Define the range of k1 (kM) parameters
k1_values = [0.0075, 0.01, 0.0125, 0.015, 0.02, 0.025]

# --- Run simulations for each k1 value ---

for k1 in k1_values:
    # Create a unique output directory for each k1 value
    output_dir = base_output_dir / f"k1_{k1}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up the exercise model. Assigns baseline parameters.
    exercise_model = ExerciseModel(protocol, exercise_config, output_dir)

    # Update the k1 / kM parameter
    exercise_model.params["k1"] = k1

    # Set up the material model
    material_model = MuscleRohrle(
        geometry.domain,
        geometry.ft,
        config_file,
        geometry.fibers,
        output_dir,
        clamp_type="robin",
    )

    # Set up the coupled model
    coupled_model = MuscleGrowthModel(
        exercise_model,
        material_model,
        output_dir,
        feedback=False,
        csa_function=geometry.compute_csa,
    )

    # Set log level
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

    # Run the simulation
    coupled_model.simulate()


# --- Plotting the results ---
plt.style.use("science")
plt.figure(figsize=(4, 3))

for k1 in k1_values:
    # Load growth results for the current k1 case
    output_dir = base_output_dir / f"k1_{k1}"
    growth_results = pd.read_csv(output_dir / "growth_results.csv")

    # Plot normalized CSA over time
    plt.plot(
        growth_results.t / 24 / 7,  # Convert to weeks
        growth_results.csa / growth_results.csa[0],
        label=rf"$k_M$={k1}",
    )

data_defreitas = pd.read_csv(Path(__file__).parents[1] / "data/defreitas.csv")
data_damas = pd.read_csv(Path(__file__).parents[1] / "data/damas.csv")
data_seynnes = pd.read_csv(Path(__file__).parents[1] / "data/seynnes.csv")

plt.plot(
    (data_seynnes["week"]),
    data_seynnes["csa"] / data_seynnes["csa"][0],
    "P",
    color="lightgray",
    label="Seynnes et al, 2007",
)
plt.plot(
    (data_damas["week"]),
    data_damas["csa"] / data_damas["csa"][0],
    "o",
    color="gray",
    label="Damas et al, 2016",
)
plt.plot(
    (data_defreitas["week"]),
    data_defreitas["csa"] / data_defreitas["csa"][0],
    "d",
    color="darkgray",
    label="DeFreitas et al, 2011",
)

# Add labels, legend, and save the combined plot
plt.ylabel("Normalized CSA")
plt.xlabel("Time (weeks)")
plt.legend()

# Adjust style

# Hide the top and right spines
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)

# Remove ticks from top and right
plt.gca().tick_params(top=False, right=False)

# Remove minor ticks
plt.gca().minorticks_off()

plt.xticks(range(0, 11))

plt.tight_layout()
plt.savefig(base_output_dir / "parameter_fit_km_defreitas.png", dpi=300)
print(f"Plot saved to {base_output_dir / 'parameter_fit_km_defreitas.png'}")
