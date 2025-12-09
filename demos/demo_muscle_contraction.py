"""
Demo Script: 3D Muscle Contraction Simulation using Hyperelastic Material Model
"""

from pathlib import Path
import numpy as np
import dolfinx

import musclex
from musclex.material import MuscleRohrle
from musclex.postprocess_mechanics import postprocess_from_file

# Set FEniCSx log level
# Set to INFO or DEBUG for more detailed logs
dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

# 1. Setup
# Define paths relative to the script location
root_dir = Path(__file__).parents[1]
config_path = root_dir / "config_files/material_rohrle.yml"
mesh_path = root_dir / "meshes/muscle-idealized/muscle-idealized.xdmf"
results_dir = root_dir / "results/demos/contraction_demo"

# Create output directory
results_dir.mkdir(parents=True, exist_ok=True)

print(f"\n{'='*60}")
print(f"{'3D MUSCLE CONTRACTION SIMULATION':^60}")
print(f"{'='*60}")

# 2. Load Geometry
print(f"Loading mesh from: {mesh_path.name}...")

# The Geometry class handles mesh loading and fiber field initialization
geometry = musclex.geometry.IdealizedFusiform(mesh_path)
geometry.info()

# 3. Initialize Material Model
# We use the transversely isotropic hyperelastic model
print(f"Initializing material model from: {config_path.name}...")

material_model = MuscleRohrle(
    geometry.domain,  # the mesh
    geometry.ft,  # facet tags marking boundaries for BCs
    config_path,  # material configuration file
    geometry.fibers,  # fiber orientation field
    results_dir,  # output directory
    clamp_type="robin",  # spring-like boundary condition at the ends
)

# 4. Run Simulation
# Ramp activation level (alpha) from 0.0 (passive) to 1.0 (fully active)
steps = 21
alphas = np.linspace(0, 1.0, steps)
np.save(results_dir / "activation_levels.npy", alphas)  # needed for postprocessing

print(f"Starting quasi-static solve for {steps} steps (alpha 0.0 -> 1.0)...")

# The solver handles the nonlinear Newton iterations for each load step
converged, _ = material_model.solve(alphas)

print(f"Simulation {'converged' if converged else 'failed'}.")

# 5. Post-Processing
print("\n--- Starting 3D Post-Processing ---")

# Pyvista-based visualization
# Generates videos of displacement, fiber stretch, and stress fields
postprocess_from_file(
    output_dir=results_dir,
    conf_file=config_path,
)

print("Done.")
