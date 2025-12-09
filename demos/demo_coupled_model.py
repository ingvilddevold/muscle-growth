"""
Demo Script: Full Coupled Muscle Growth Simulation
Integrates Signaling (ODEs) + Tissue Mechanics (Hyperelasticity) + Volumetric Growth.

This simulation takes around 5 minutes to run on a standard laptop.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import dolfinx
import pyvista

from musclex.geometry import IdealizedFusiform
from musclex.material import MuscleRohrle
from musclex.protocol import RegularExercise
from musclex.exercise_model import ExerciseModel
from musclex.muscle_growth import MuscleGrowthModel
from musclex.postprocess_growth import postprocess_from_file

# Set FEniCSx log level
dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

# 1. Setup
root_dir = Path(__file__).resolve().parents[1]
results_dir = root_dir / "results/demos/coupled_growth"
results_dir.mkdir(parents=True, exist_ok=True)

mesh_path = root_dir / "meshes/muscle-idealized/muscle-idealized.xdmf"
material_config = root_dir / "config_files/material_rohrle.yml"
exercise_config = root_dir / "config_files/exercise_eq_reduced_k1.yml"

print(f"{'='*60}")
print(f"{'FULL COUPLED MUSCLE GROWTH SIMULATION':^60}")
print(f"{'='*60}")

# 2. Geometry Setup
print(f"Loading geometry from: {mesh_path.name}")
# The Geometry class handles mesh loading and fiber field initialization
geometry = IdealizedFusiform(mesh_path)
geometry.info()

# Visualize fibers
pyvista.OFF_SCREEN = True
plotter = geometry.plot(mode="fibers")
plotter.view_yz()
plotter.camera.zoom(1.5)
screenshot_path = results_dir / "geometry_fibers.png"
plotter.screenshot(screenshot_path, scale=2)
print(f"Saved geometry screenshot to: {screenshot_path.name}")

# 3. Material Model Setup (Tissue Mechanics)
# We use the transversely isotropic hyperelastic model
print(f"Initializing material model...")
material_model = MuscleRohrle(
    geometry.domain,  #  mesh
    geometry.ft,  # facet tags for BCs
    material_config,  # material configuration file
    geometry.fibers,  # fiber orientation field
    results_dir,  # output directory
    clamp_type="robin",  # spring-like BC at ends
)

# 4. Exercise Model Setup (Signaling)
print(f"Initializing signaling model...")
# Protocol: 1 hour exercise bout every 2 days for 7 bouts / 15 days
protocol = RegularExercise(
    N=7, exercise_duration=1, growth_duration=47, end_time=15 * 24, intensity=0.8
)
exercise_model = ExerciseModel(protocol, exercise_config, results_dir)

# 5. Coupled Model Setup
# Combining Exercise Model + Material Model through the volumetric growth framework
print("Initializing coupled growth model...")
coupled_model = MuscleGrowthModel(
    exercise_model, material_model, results_dir, geometry.compute_csa
)

# 6. Run Simulation
print(f"Starting simulation for {protocol.events[-1].end_time/24:.1f} days...")
coupled_model.simulate()

# Save final signaling state events
coupled_model.exercise_model.assemble_solution()
ode_df = coupled_model.exercise_model.solution_dataframe()
ode_df.to_csv(results_dir / "ode_results.csv")
print("Simulation complete.")

# 7. Post-Processing / Visualization with PyVista
print("\n--- Starting Post-Processing ---")
postprocess_from_file(
    output_dir=results_dir,
    conf_file=material_config,
    freq=2,
    warp_scale=10.0,
    camera_pos="idealized-fusiform",
)

# 8. Plotting Results
print("\nGenerating result plots...")
ode_results = pd.read_csv(results_dir / "ode_results.csv")
growth_results = pd.read_csv(results_dir / "growth_results.csv")

t_days = ode_results.t / 24
t_growth = growth_results.t / 24
params = exercise_model.params

# Configuration: (x_data, y_data, color, label, ylabel, threshold_value)
plot_cfg = [
    (t_days, ode_results.igf1, "#D8CC39", "IGF1", "IGF1", None),
    (t_days, ode_results.akt, "#46A0D4", "AKT", "AKT", None),
    (t_days, ode_results.foxo, "#BF4A00", "FOXO", "FOXO", params["x3_th"]),
    (t_days, ode_results.mtor, "#00885F", "mTOR", "mTOR", params["x4_th"]),
    (t_growth, growth_results.k1, "black", r"$k_M$", r"$k_M$ ($h^{-1}$)", None),
    (t_growth, growth_results.csa / growth_results.csa[0], "purple", "CSA", "Norm. CSA", None),
    (t_growth, growth_results.volume / growth_results.volume[0], "navy", "Volume", "Norm. Vol", None),
]

fig, axs = plt.subplots(len(plot_cfg), 1, figsize=(7, 12), sharex=True)

for ax, (x, y, color, label, ylabel, thresh) in zip(axs, plot_cfg):
    ax.plot(x, y, color=color, label=label)
    if thresh:
        ax.axhline(thresh, color="gray", linestyle=":", alpha=0.7, label="Threshold")
    
    loc = "lower right" if any(x in label for x in ["mTOR", "CSA", "Vol"]) else "upper right"
    ax.legend(loc=loc, fontsize="small")
    
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, linestyle=":", alpha=0.4)

axs[-1].set_xlabel("Time (days)", fontweight="bold")
plt.tight_layout()
plt.savefig(results_dir / "summary_results.png", dpi=300)
plt.close()

print(f"Done. Results saved to: {results_dir}")
