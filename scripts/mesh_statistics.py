import dolfinx
import numpy as np
import pandas as pd
from mpi4py import MPI
from musclex.geometry import RealisticGeometry
from pathlib import Path
import argparse

def main(mesh_files, output_file):
    """
    Processes a list of mesh files to compute and save geometric statistics.
    """
    all_stats = []
    for mf in mesh_files:
        mesh_path = Path(mf)
        print(f"Processing mesh file: {mesh_path}")

        # Construct muscle geometry from files
        meshfile = mesh_path / f"{mesh_path.stem}.xdmf"
        fibersfile = mesh_path / f"{mesh_path.stem}_fibers.bp"
        geometry = RealisticGeometry(meshfile, fibersfile=fibersfile, comm=MPI.COMM_WORLD)
        geometry.setup_csa_surface(method="box", resolution=30, thickness=1e-3)

        volume = geometry.volume
        tdim = geometry.domain.topology.dim
        num_vertices = geometry.domain.geometry.x.shape[0]
        num_cells = geometry.domain.topology.index_map(tdim).size_local
        h = dolfinx.cpp.mesh.h(geometry.domain._cpp_object, tdim, np.arange(num_cells))
        h_max = h.max()
        h_min = h.min()
        csa = geometry.compute_csa()

        # Collect statistics for this mesh
        stats = {
            "mesh": mesh_path.stem,
            "num_vertices": num_vertices,
            "num_cells": num_cells,
            "volume_cm3": volume * 1e6,  # Convert m^3 to cm^3
            "h_max_cm": h_max * 1e2,     # Convert m to cm
            "h_min_cm": h_min * 1e2,     # Convert m to cm
            "csa_cm2": csa * 1e4,        # Convert m^2 to cm^2
        }
        all_stats.append(stats)

    # Write all statistics to a CSV file
    df = pd.DataFrame(all_stats)
    df.to_csv(output_file, index=False)
    print(f"Mesh statistics written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute mesh statistics for Dolfinx models.")
    
    parser.add_argument(
        "--mesh-files", 
        type=str, 
        required=True,
        help="Comma-separated list of paths to the mesh directories."
    )
    parser.add_argument(
        "--output-file", 
        type=str, 
        required=True,
        help="Path to the output CSV file."
    )

    args = parser.parse_args()
    
    # Split the comma-separated string of mesh files into a list
    mesh_files_list = args.mesh_files.split(',')

    main(mesh_files_list, args.output_file)
