import numpy as np
import pandas as pd
from pathlib import Path
import multiprocessing as mp
import time
import os, sys
import yaml
from tqdm import tqdm
import matplotlib.pyplot as plt

from SALib.analyze.sobol import analyze
from SALib.sample.sobol import sample

from musclex.protocol import RegularExercise
from musclex.exercise_model import ExerciseModel

# Prepare for server-side plotting
import matplotlib

matplotlib.use("Agg")
# disable latex
plt.rcParams["text.usetex"] = False


# Prepare the baseline exercise model
protocol = RegularExercise(15, 1, 23, 20 * 24)
exercise_config = Path(__file__).parents[1] / "config_files/exercise_eq.yml"
# load parameter values from yml
with open(exercise_config, "r") as f:
    p = yaml.safe_load(f)["exercise_parameters"]

# Names of parameters included in sensitivity analysis
# These should match the keys in the p dictionary from the config file
names_sa = [
    "a10",
    "a20",
    "a3",
    "a4",
    "b1",
    "b2",
    "b3",
    "b4",
    "c21",
    "c32",
    "c42",
    "c43",
    "k1",
    "k2",
]

# Define the problem for sensitivity analysis
# We test parameters in a range around the baseline values, from 50% to 200% of the baseline values
lower_bound_multiplier = 0.5
upper_bound_multiplier = 2
problem = {
    "num_vars": len(names_sa),
    "names": names_sa,
    "bounds": [
        [p["a10"] * lower_bound_multiplier, p["a10"] * upper_bound_multiplier],  # a10
        [p["a20"] * lower_bound_multiplier, p["a20"] * upper_bound_multiplier],  # a20
        [p["a3"] * lower_bound_multiplier, p["a3"] * upper_bound_multiplier],  # a3
        [p["a4"] * lower_bound_multiplier, p["a4"] * upper_bound_multiplier],  # a4
        [p["b1"] * lower_bound_multiplier, p["b1"] * upper_bound_multiplier],  # b1
        [p["b2"] * lower_bound_multiplier, p["b2"] * upper_bound_multiplier],  # b2
        [p["b3"] * lower_bound_multiplier, p["b3"] * upper_bound_multiplier],  # b3
        [p["b4"] * lower_bound_multiplier, p["b4"] * upper_bound_multiplier],  # b4
        [p["c21"] * lower_bound_multiplier, p["c21"] * upper_bound_multiplier],  # c21
        [p["c32"] * lower_bound_multiplier, p["c32"] * upper_bound_multiplier],  # c32
        [p["c42"] * lower_bound_multiplier, p["c42"] * upper_bound_multiplier],  # c42
        [p["c43"] * lower_bound_multiplier, p["c43"] * upper_bound_multiplier],  # c43
        [p["k1"] * lower_bound_multiplier, p["k1"] * upper_bound_multiplier],  # k1
        [p["k2"] * lower_bound_multiplier, p["k2"] * upper_bound_multiplier],  # k2
    ],
}

# Prepare parallel processing
N_samples = 10000
print(f"Using N={N_samples}, num_vars={problem['num_vars']}")

try:
    # Get CPU count allocated by SLURM
    num_workers = int(os.environ.get("SLURM_CPUS_PER_TASK", mp.cpu_count()))
    print(f"Using {num_workers} workers as specified by SLURM_CPUS_PER_TASK.")
except TypeError:
    # Fallback if SLURM_CPUS_PER_TASK is not set (e.g., running locally)
    num_workers = mp.cpu_count()
    print(
        f"SLURM_CPUS_PER_TASK not set. Using {num_workers} workers based on available CPU cores."
    )


# Set up the output directory
output_dir = (
    Path(__file__).parents[1]
    / "results/SA"
    / f"sensitivity_analysis_parallel_N={N_samples}_order=2"
)
output_dir.mkdir(parents=True, exist_ok=True)


def evaluate_model_for_params(params):
    """
    Runs the ExerciseModel for a given parameter set. Returns the output (number of myofibrils z)
    """
    # Construct a new exercise model
    model = ExerciseModel(protocol, exercise_config, results_path=None)

    # Update the parameters from the sample
    for name, value in zip(problem["names"], params):
        model.params[name] = value

    # Run the simulation
    model.simulate()

    # Extract the result (number of myofibrils z at the last time step)
    z = model.y[4, -1]
    return z


if __name__ == "__main__":

    # Generate samples
    print("Generating parameter samples...")
    param_values = sample(problem, N_samples, calc_second_order=True)
    num_simulations = len(param_values)
    print(f"Generated {num_simulations} parameter sets for Sobol.")

    # Run simulations in parallel
    print(f"Running simulations in parallel using {num_workers} workers...")
    start_time = time.time()
    results_list = []

    with mp.Pool(processes=num_workers) as pool:
        # Use imap to get results in order, wrapped with tqdm for progress
        result_iterator = pool.imap(evaluate_model_for_params, param_values)

        # Iterate through the results using tqdm progress bar
        for result in tqdm(
            result_iterator,
            total=num_simulations,
            desc="Simulations",
            unit="run",
            file=sys.stdout,
            mininterval=1.0,
        ):
            results_list.append(result)
            sys.stdout.flush()

    end_time = time.time()
    print(f"\nSimulations finished in {end_time - start_time:.2f} seconds.")

    # Convert results list to np.array (order preserved by imap)
    Y = np.array(results_list)

    # Save results to CSV
    results_file = output_dir / "model_outputs.csv"
    with results_file.open("w") as f:
        # Write header
        f.write("index," + ",".join(problem["names"]) + ",z\n")

        # Write each parameter set and its corresponding result
        for i, (params, z) in enumerate(zip(param_values, Y)):
            f.write(f"{i}," + ",".join(map(str, params)) + f",{z}\n")

    results_file = output_dir / "model_outputs.csv"
    # Read results into a DataFrame
    results_df = pd.read_csv(results_file)
    Y = np.array(results_df["z"])

    # Analyze results
    print("Analyzing results...")
    Si = analyze(problem, Y, calc_second_order=True, print_to_console=True)

    # Get the parameter names from your problem definition
    param_names = problem["names"]

    # Convert to dataframe
    Si_df = Si.to_df()

    # Save to CSV files
    Si_df[0].to_csv(output_dir / "ST.csv")
    Si_df[1].to_csv(output_dir / "S1.csv")
    Si_df[2].to_csv(output_dir / "S2.csv")
    print("SA results saved to:", output_dir)

    print("\n--- Sensitivity indices ---")
    print(Si)

    Si.plot()
    fig = plt.gcf()
    fig.set_size_inches(10, 4)

    # save the plot
    plt.savefig(output_dir / "sensitivity_indices.png", dpi=300)
