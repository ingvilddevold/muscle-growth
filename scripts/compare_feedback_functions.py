from pathlib import Path

import dolfinx

import musclex
from musclex.exercise_model import ExerciseModel
from musclex.material import MuscleRohrle
from musclex.muscle_growth import MuscleGrowthModel
from musclex.protocol import RegularExercise

feedback_functions = {"linear": "Linear", "hill": "Hill", "none": "No feedback"}

# Base output directory
base_output_dir = (
    Path(__file__).parents[1]
    / "results"
    / "coupled"
    / "feedback_functions_comparison"
)

# Define the exercise protocol
protocol = RegularExercise(15, 1, 23, 20 * 24)

# Loop over feedback types
for feedback_type in feedback_functions.keys():

    # Create a unique output directory for each feedback type
    output_dir = base_output_dir / f"feedback_{feedback_type}"
    output_dir.mkdir(exist_ok=True, parents=True)

    # Set up the exercise model
    exercise_config = (
        Path(__file__).parents[1] / "config_files/exercise_eq_reduced_k1.yml"
    )
    exercise_model = ExerciseModel(protocol, exercise_config, output_dir)

    # Set up the material model
    geometry = musclex.geometry.CylinderGmsh()
    config_file = Path(__file__).parents[1] / "config_files/material_rohrle.yml"
    material_model = MuscleRohrle(
        geometry.domain,
        geometry.ft,
        config_file,
        geometry.fibers,
        output_dir,
    )

    # Set up the coupled model
    if feedback_type == "none":
        feedback_type = False  # Disable feedback
    coupled_model = MuscleGrowthModel(
        exercise_model,
        material_model,
        output_dir,
        feedback=feedback_type,
        csa_function=geometry.compute_csa,
    )

    # Set log level
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

    # Run the simulation
    coupled_model.simulate()


# Read and plot results for all feedback cases
import pandas as pd
import scienceplots
from matplotlib import pyplot as plt

plt.style.use(["science"])

plt.figure(figsize=(3, 2))

for feedback_type, feedback_name in feedback_functions.items():
    # Load growth results for the current k1 case
    output_dir = base_output_dir / f"feedback_{feedback_type}"
    growth_results = pd.read_csv(output_dir / "growth_results.csv")

    # Set line style to : for last feedback type
    print(f"Plotting results for feedback type: {feedback_type}")
    ls = ":" if feedback_type == "hill" else "-"
    print(f"Line style: {ls}")

    # Plot normalized CSA over time
    plt.plot(
        growth_results.t / 24,
        growth_results.csa / growth_results.csa[0],
        label=f"{feedback_name}",
        linestyle=ls,
    )

plt.ylabel("Normalized CSA")
plt.xlabel("Time (days)")
plt.legend()
# Adjust
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().tick_params(top=False, right=False)
plt.gca().minorticks_off()
plt.tight_layout()
# Save
plt.savefig(base_output_dir / "feedback_comparison_CSA.png", dpi=400)


plt.figure(figsize=(3, 2))

for feedback_type, feedback_name in feedback_functions.items():
    # Load growth results for the current k1 case
    output_dir = base_output_dir / f"feedback_{feedback_type}"
    growth_results = pd.read_csv(output_dir / "growth_results.csv")

    # Set line style to : for last feedback type
    print(f"Plotting results for feedback type: {feedback_type}")
    ls = ":" if feedback_type == "hill" else "-"
    print(f"Line style: {ls}")

    # Plot normalized CSA over time
    plt.plot(
        growth_results.t / 24, growth_results.k1, label=f"{feedback_name}", linestyle=ls
    )

plt.ylabel(r"Protein synthesis rate $k_M$")
plt.xlabel("Time (days)")
plt.legend()
# Adjust
plt.gca().spines["top"].set_visible(False)
plt.gca().spines["right"].set_visible(False)
plt.gca().tick_params(top=False, right=False)
plt.gca().minorticks_off()
plt.tight_layout()
# Save
plt.savefig(base_output_dir / "feedback_comparison_k1.png", dpi=400)


# Make a combined plot with both CSA and k1 from saved png files
from PIL import Image

# Load the images
csa_image = Image.open(base_output_dir / "feedback_comparison_CSA.png")
k1_image = Image.open(base_output_dir / "feedback_comparison_k1.png")
# Create a new image with double the width
combined_image = Image.new(
    "RGB", (csa_image.width + k1_image.width, csa_image.height), (255, 255, 255)
)
# Paste the images into the combined image
combined_image.paste(k1_image, (0, 0))
combined_image.paste(csa_image, (k1_image.width, 0))
# Save the combined image
combined_image.save(
    base_output_dir / "feedback_comparison_combined.png", dpi=(400, 400)
)
