"""
This script runs a single coupled muscle growth simulation for a given
exercise protocol. It is designed to be called by Snakemake.
"""

import dolfinx
from mpi4py import MPI
from musclex.protocol import DeFreitasProtocol, RegularExercise
from musclex.exercise_model import ExerciseModel
from musclex.muscle_growth import MuscleGrowthModel
from musclex.material import MuscleRohrle
from musclex.geometry import IdealizedFusiform, RealisticGeometry
from pathlib import Path
import typer


protocols = {
    "testing": RegularExercise(
        N=1, exercise_duration=1, growth_duration=23, end_time=27
    ),
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


def main(
    protocol_name: str = typer.Option(
        ..., help="Name of the exercise protocol to run."
    ),
    output_dir: Path = typer.Option(..., help="Directory to save simulation output."),
    exercise_config: Path = typer.Option(
        ..., help="Path to the exercise model config file."
    ),
    material_config: Path = typer.Option(
        ..., help="Path to the material model config file."
    ),
    mesh_path: Path = typer.Option(
        ..., help="Path to the directory containing mesh files."
    ),
    output_freq: int = typer.Option(1, help="Frequency for saving output files."),
    is_realistic: bool = typer.Option(
        False, help="Flag for realistic vs. idealized geometry."
    ),
    is_spatial: bool = typer.Option(
        False, help="Flag for spatial vs. non-spatial exercise model."
    ),
    variation_magnitude: float = typer.Option(
        0.01, help="Magnitude of spatial variation."
    ),
    seed: int = typer.Option(42, help="Random seed for spatial variation."),
):
    """
    Runs a single coupled muscle growth simulation.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    protocol = protocols[protocol_name]

    output_dir.mkdir(exist_ok=True, parents=True)
    if rank == 0:
        print(f"Running coupled growth for protocol: {protocol_name}")
        print(f"Output will be saved to: {output_dir}")

    # --- Initialize Geometry ---
    if is_realistic:
        meshfile = mesh_path / (f"{mesh_path.stem}.xdmf")
        fibersfile = mesh_path / (f"{mesh_path.stem}_fibers.bp")
        geometry = RealisticGeometry(
            meshfile, fibersfile=fibersfile, comm=MPI.COMM_WORLD
        )
        geometry.setup_csa_surface(method="box", resolution=30, thickness=1e-3)
    else:
        meshfile = mesh_path / (f"{mesh_path.stem}.xdmf")
        geometry = IdealizedFusiform(meshfile, comm=MPI.COMM_WORLD)

    # --- Initialize Exercise Model ---
    if is_spatial:
        from musclex.exercise_model_spatial import SpatialExerciseModel

        exercise_model = SpatialExerciseModel(
            geometry.domain,
            protocol,
            exercise_config,
            output_dir,
            variation_magnitude,
            seed=seed,
        )
    else:
        exercise_model = ExerciseModel(protocol, exercise_config, output_dir)

    # --- Initialize Material Model ---
    material_model = MuscleRohrle(
        geometry.domain,
        geometry.ft,
        material_config,
        geometry.fibers,
        output_dir,
        clamp_type="robin",
    )

    # --- Initialize Coupled Growth Model ---
    coupled_model = MuscleGrowthModel(
        exercise_model,
        material_model,
        output_dir,
        output_freq=output_freq,
        csa_function=geometry.compute_csa,
    )

    # --- Run simulation ---
    dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)
    coupled_model.simulate()

    if rank == 0:
        print(f"Simulation for {protocol_name} complete.")


if __name__ == "__main__":
    typer.run(main)
