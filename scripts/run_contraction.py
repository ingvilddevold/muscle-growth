import numpy as np
import pandas as pd
from pathlib import Path
import typer

import dolfinx
from musclex.material import MuscleRohrle
from musclex.geometry import IdealizedFusiform, RealisticGeometry


app = typer.Typer()

@app.command()
def main(
    output_dir: Path = typer.Option(..., "--output-dir", help="The directory to save the simulation results."),
    material_config_path: Path = typer.Option(..., "--material-config", help="Path to the material configuration YAML file."),
    mesh_name: str = typer.Option(..., "--mesh-name", help="Mesh name."),
    mesh_path: Path = typer.Option(..., "--mesh-path", help="Path to the directory containing the mesh files."),
    is_realistic: bool = typer.Option(True, "--is-realistic/--no-is-realistic", help="Flag to indicate if the mesh is realistic or idealized."),
):
    """
    Runs an active contraction simulation for a given muscle geometry.
    """
    # Create the output directory if it doesn't exist
    output_dir.mkdir(exist_ok=True, parents=True)
    print(f"Running active contraction for mesh: '{mesh_name}'")
    print(f"Results will be saved to: {output_dir}")

    # Reduce verbosity during setup
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

    # --- 2. Set up the simulation ---
    # Load the geometry and fiber directions from the mesh file
    if is_realistic:
        meshfile = mesh_path / f"{mesh_path.stem}.xdmf"
        fibersfile = mesh_path / f"{mesh_path.stem}_fibers.bp"
        geometry = RealisticGeometry(meshfile, fibersfile=fibersfile)
    else:
        meshfile = mesh_path / f"{mesh_path.stem}.xdmf"
        geometry = IdealizedFusiform(meshfile)

    # Initialize the material model. The boundary conditions and problem
    # setup are handled internally by this class.
    material_model = MuscleRohrle(
        geometry.domain,
        geometry.ft,
        material_config_path,
        geometry.fibers,
        output_dir,
        clamp_type="robin"
    )

    # --- 3. Run the simulation ---
    # Define the activation levels to test
    activation_levels = np.linspace(0.0, 1.0, 100)

    print("Starting simulation using material_model.solve()...")
    converged, forces = material_model.solve(activation_levels)

    if converged:
        print("Simulation complete.")
    else:
        print("Warning: Simulation did not converge.")

    # --- 4. Save the results ---
    # Create a dictionary for the DataFrame
    print("len(forces): ", len(forces))
    print("len(activation_levels): ", len(activation_levels))
    results_data = {
        "activation": activation_levels[:len(forces)],
        "force": forces
    }
    results_df = pd.DataFrame(results_data)

    # Define the output file path
    output_csv = output_dir / f"force_activation_{mesh_name}.csv"

    # Save the DataFrame to a CSV file
    results_df.to_csv(output_csv, index=False)

    # Save activation levels as a numpy file for reference
    output_npy = output_dir / "activation_levels.npy"
    np.save(output_npy, activation_levels[:len(forces)])

    print(f"Force-activation results saved successfully to {output_csv}")
    print("\n--- Final Results ---")
    print(results_df)


if __name__ == "__main__":
    app()
