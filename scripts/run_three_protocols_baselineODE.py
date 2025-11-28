"""
This script runs ODE exercise model simulations for three different exercise
protocols and saves the ODE solutions to CSV files. It is intended to be called
from a Snakemake rule.
"""

from musclex.protocol import DeFreitasProtocol, RegularExercise
from musclex.exercise_model import ExerciseModel
from pathlib import Path

# --- Protocol Definitions ---
protocols = {
    "defreitas": DeFreitasProtocol(),
    "weekly": RegularExercise(
        N=8,
        exercise_duration=1,
        growth_duration=23 + 24 * 6,
        end_time=9 * 7 * 24,
        initial_rest=7 * 24,
    ),
    "everythreedays": RegularExercise(
        N=18,
        exercise_duration=4,
        growth_duration=20 + 24 * 2,
        end_time=9 * 7 * 24,
        initial_rest=7 * 24,
        intensity=0.2,
    ),
}

# --- Snakemake Integration ---
config_file = snakemake.input.config
output_files = snakemake.output
protocol_names = snakemake.params.protocols

print("Starting simulations for three protocols...")

# --- Simulation Loop ---
# Iterate through the protocols, run the simulation, and save the output.
for i, protocol_name in enumerate(protocol_names):
    print(f"  Running protocol: {protocol_name}...")

    protocol = protocols[protocol_name]
    output_csv = Path(output_files[i])

    # Ensure the directory for the output file exists.
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    # Instantiate the model with the given protocol and config file.
    model = ExerciseModel(
        protocol,
        str(config_file),
    )

    # Run the simulation.
    model.simulate()

    # Get the results and save to the specified CSV file.
    ode_results_df = model.solution_dataframe()
    ode_results_df.to_csv(output_csv)

    print(f"  Successfully saved results to '{output_csv}'")

print("All simulations complete.")
