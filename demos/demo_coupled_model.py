# %% [markdown]
# # Simulation of muscle growth with coupled signaling-mechanics model
#
# This demo integrates the **IGF1-AKT-mTOR-FOXO signaling model** with **3D hyperelastic tissue mechanics** to simulate exercise-induced hypertrophy.
#
# The coupling works in two directions:
# 1.  **Signaling $\to$ Mechanics:** The net protein balance ($N$) from the signaling model drives a volumetric **growth tensor** ($\mathbf{G}$), which expands the tissue perpendicular to the muscle fibers.
# 2.  **Mechanics $\to$ Signaling:** The change in muscle Cross-Sectional Area (CSA) modulates the protein synthesis rate ($k_M$), creating a homeostatic loop that limits unbounded growth.
#
# We will simulate a 15-day training protocol on an idealized fusiform muscle geometry to keep run time reasonable.

# %%
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import dolfinx
import pyvista as pv

from musclex.geometry import IdealizedFusiform
from musclex.material import MuscleRohrle
from musclex.protocol import RegularExercise
from musclex.exercise_model import ExerciseModel
from musclex.muscle_growth import MuscleGrowthModel
from musclex.postprocess_growth import postprocess_from_file

# Set FEniCSx log level
dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

# %% [markdown]
# ## Setup
# We define the input mesh paths and configuration files for the material properties and signaling parameters.

# %%
root_dir = Path("..")
results_dir = root_dir / "results/demos/coupled_growth"
results_dir.mkdir(parents=True, exist_ok=True)

mesh_path = root_dir / "meshes/muscle-idealized/muscle-idealized.xdmf"
material_config = root_dir / "config_files/material_rohrle.yml"
exercise_config = root_dir / "config_files/exercise_eq_reduced_k1.yml"

# %% [markdown]
# ## Load geometry and fiber architecture
# We use an idealized fusiform muscle. The fiber direction dictates the orientation of growth, with the muscle expanding perpendicular to the fibers.
#

# %%
geometry = IdealizedFusiform(mesh_path)
geometry.info()

# %%
# Visualize geometry
pv.set_jupyter_backend("static")
plotter = geometry.plot(mode="all")

# %% [markdown]
# ## Initialize material and signaling models
# We initialize the two sub-models independently before coupling them:
# * **Material Model:** Transversely isotropic hyperelasticity with tendon-like (Robin) boundary conditions.
# * **Exercise Model:** Modeling the IGF1-AKT-mTOR-FOXO signaling pathway. The exercise protocol consists of 1h exercise bouts every 2 days for 15 days.

# %%
print(f"Initializing material model...")
material_model = MuscleRohrle(
    geometry.domain,     # mesh
    geometry.ft,         # facet tags for BCs
    material_config,     # material configuration file
    geometry.fibers,     # fiber orientation field
    results_dir,         # output directory
    clamp_type="robin",  # spring-like BC at ends
)

print(f"Initializing signaling model...")
# Protocol: 1 hour exercise bout every 2 days (48h cycle) for 15 days
protocol = RegularExercise(
    N=7, exercise_duration=1, growth_duration=47, end_time=15 * 24, intensity=0.8
)
exercise_model = ExerciseModel(protocol, exercise_config, results_dir)

# %% [markdown]
# ## Configure coupled growth model
# The `MuscleGrowthModel` class manages the model coupling. At each step, it:
# 1.  Advances the ODEs (Signaling)
# 2.  Updates the growth tensor $\mathbf{G}$.
# 3.  Solves for mechanical equilibrium (hyperelasticity).
# 4.  Computes the new CSA and updates $k_M$ via feedback.

# %%
print("Initializing coupled growth model...")
coupled_model = MuscleGrowthModel(
    exercise_model, material_model, results_dir, geometry.compute_csa
)

# %% [markdown]
# ## Run simulation
# This process iterates through the exercise and rest periods defined in the protocol.
#
# *Note: This may take a few (5-10) minutes*

# %% tags=["output_scroll"]
print(f"Starting simulation for {protocol.events[-1].end_time/24:.1f} days...")
coupled_model.simulate()

# Save signaling states
coupled_model.exercise_model.assemble_solution()
ode_df = coupled_model.exercise_model.solution_dataframe()
ode_df.to_csv(results_dir / "ode_results.csv")
print("Simulation complete.")

# %% [markdown]
# ## Post-processing and visualization
# We generate a summary plot showing the dynamics of all system variables:
# * The signaling states (IGF1, AKT, FOXO, mTOR, myofibrils N)
# * The homeostatic feedback ($k_M$ decreases as the muscle grows)
# * The macroscopic hypertrophy (CSA and Volume increasing over time)

# %%
ode_results = pd.read_csv(results_dir / "ode_results.csv")
growth_results = pd.read_csv(results_dir / "growth_results.csv")
params = exercise_model.params

t_days = ode_results.t / 24
t_growth = growth_results.t / 24

# Normalize CSA and Volume
csa_norm = growth_results.csa / growth_results.csa[0]
volume_norm = growth_results.volume / growth_results.volume[0]

# Configuration: (x_data, y_data, color, label, ylabel, threshold_value)
plot_cfg = [
    (t_days, ode_results.igf1, "#D8CC39", "IGF1", "IGF1", None),
    (t_days, ode_results.akt, "#46A0D4", "AKT", "AKT", None),
    (t_days, ode_results.foxo, "#BF4A00", "FOXO", "FOXO", params["x3_th"]),
    (t_days, ode_results.mtor, "#00885F", "mTOR", "mTOR", params["x4_th"]),
    (t_days, ode_results.z, "#882400", "Myofibrils N", "Myofibrils N", None),
    (t_growth, growth_results.k1, "black", r"$k_M$", r"$k_M$ ($h^{-1}$)", None),
    (t_growth, csa_norm, "purple", "CSA", "Norm. CSA", None),
    (t_growth, volume_norm, "navy", "Volume", "Norm. Vol", None),
]

fig, axs = plt.subplots(4, 2, figsize=(7, 7), sharex=True)

for ax, (x, y, color, label, ylabel, thresh) in zip(axs.flatten(), plot_cfg):
    ax.plot(x, y, color=color, label=label)
    if thresh:
        ax.axhline(thresh, color="gray", linestyle=":", alpha=0.7, label="Threshold")
    ax.legend()
    ax.set_xlabel("Time (days)")
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle=":", alpha=0.4)

plt.tight_layout()
plt.show()

# %% [markdown]
# We generate 3D visualizations of the muscle geometry deforming over time due to growth.
# The post-processing step creates a series of snapshots that are combined into an animation.

# %% tags=["output_scroll"]
postprocess_from_file(
    output_dir=results_dir,
    conf_file=material_config,
    freq=2,  # only process every 2nd time point
    warp_scale=10.0,  # exaggerate deformation 10x for visualization
    camera_pos="idealized-fusiform",
)

# %% [markdown]
# The animation below shows the muscle thickening over the 15-day protocol.
# The warp scale is set to 10x to better visualize the deformation.
#
# In accordance with our chosen growth tensor, the muscle thickens (increased CSA) while the overall length remains approximately constant.
# <img src="../results/demos/coupled_growth_postprocessed/video/animation_displacement.gif" width="600" align="center">
