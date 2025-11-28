import dolfinx
import ufl
import adios4dolfinx
import numpy as np
import pyvista
from pathlib import Path
from mpi4py import MPI
import typer
import imageio
import os
from musclex.utils import get_interpolation_points


# Set pyvista font and theme
pyvista.global_theme.font.family = "arial"
pyvista.global_theme.cmap = "viridis"

M2MM = 1e3  # Conversion from meters to millimeters

def get_displacment_at_time(
    bp_file: Path, mesh: dolfinx.mesh.Mesh, time: float
) -> dolfinx.fem.Function:
    """Reads a displacement function from a BP file at a specific time."""
    V = dolfinx.fem.functionspace(mesh, ("Lagrange", 2, (mesh.geometry.dim,)))
    u = dolfinx.fem.Function(V, name="u_growth")
    adios4dolfinx.read_function(bp_file, u, time=time)
    return u


def get_disp_magnitude(
    mesh: dolfinx.mesh.Mesh, u: dolfinx.fem.Function
) -> np.ndarray:
    """Computes the magnitude of a displacement field on a DG0 space."""
    DG0 = dolfinx.fem.functionspace(mesh, ("DG", 0))
    disp_mag_func = dolfinx.fem.Function(DG0)
    disp_expr = dolfinx.fem.Expression(
        ufl.sqrt(ufl.dot(u, u)), get_interpolation_points(DG0.element)
    )
    disp_mag_func.interpolate(disp_expr)
    return disp_mag_func.x.array


def main(
    mesh_file: Path = typer.Option(..., help="Path to the original XDMF mesh file."),
    sim_dirs: list[Path] = typer.Option(..., help="List of simulation output directories."),
    protocols: str = typer.Option(..., help="Comma-separated list of protocol names."),
    output_file: Path = typer.Option(..., help="Path to save the final PNG image."),
    warp_scale: float = typer.Option(10.0, help="Scaling factor for deformation."),
    camera_pos: str = typer.Option(None, help="PyVista camera position string."),
):
    """
    Generates a 3D comparison plot of muscle growth for multiple protocols.
    """
    protocol_names = protocols.split(",")
    if len(sim_dirs) != len(protocol_names):
        raise ValueError("Number of simulation directories must match number of protocols.")

    print("--- Starting 3D Growth Comparison Plot ---")
    # pyvista.start_xvfb()

    # --- 1. Load base mesh and find global color limits ---
    print("Reading base mesh...")
    mesh = adios4dolfinx.read_mesh(
        sim_dirs[0] / "u_growth_pp.bp",
        MPI.COMM_WORLD,
        engine="BP4",
        ghost_mode=dolfinx.mesh.GhostMode.none,
    )

    print("Finding final time points and global displacement limits...")
    final_times = {}
    max_disp = 0.0
    for i, sim_dir in enumerate(sim_dirs):
        protocol = protocol_names[i]
        
        # Load times from the .npy file
        times_file = sim_dir / "output_times.npy"
        if not times_file.exists():
            raise FileNotFoundError(
                f"Could not find times file for protocol '{protocol}': {times_file}"
            )
        available_times = np.load(times_file)
        last_time = available_times[-1]
        final_times[protocol] = last_time
        print(f"  - Protocol '{protocol}' final time: {last_time:.2f}")

        bp_file = sim_dir / "u_growth_pp.bp"
        u_final = get_displacment_at_time(bp_file, mesh, last_time)
        disp_mag = get_disp_magnitude(mesh, u_final)
        max_disp = max(max_disp, disp_mag.max())

    clim = [0, max_disp]
    print(f"Global displacement limits for color bar: [{clim[0]:.3f}, {clim[1]:.3f}] m")

    # --- 2. Generate individual plot frames ---
    print("\n--- Generating individual plot frames ---")
    temp_files = []
    plotter_kwargs = dict(
        window_size=[300, 500],  # Size for a single vertical plot
        off_screen=True,
        border=False,
    )

    # --- Plot 0: Ungrown ---
    print("  - Plotting Ungrown muscle")
    plotter = pyvista.Plotter(**plotter_kwargs)
    plotter.add_text("Ungrown", font_size=12, position="upper_edge")
    topo, cells, geom = dolfinx.plot.vtk_mesh(mesh)
    grid = pyvista.UnstructuredGrid(topo, cells, geom)
    grid.cell_data["Displacement (mm)"] = np.zeros(grid.n_cells)
    plotter.add_mesh(grid, scalars="Displacement (mm)", show_edges=False, clim=clim, show_scalar_bar=False)
    if camera_pos:
        plotter.camera_position = eval(camera_pos)
    else:
        plotter.view_yz()
        camera_position = plotter.camera_position  # Save for later use
    ungrown_file = output_file.parent / "_temp_0_ungrown.png"
    plotter.screenshot(ungrown_file, scale=4, transparent_background=True)
    temp_files.append(ungrown_file)
    plotter.close()

    # --- Plots 1-3: Grown protocols ---
    sargs = dict(
        title="", #Displacement (mm)",
        vertical=True,
        title_font_size=22,
        label_font_size=20,
        shadow=False,
        n_labels=2,
        fmt="%.2f",
        #height=0.5,
        #width=0.2,
        position_x=0.75,
        position_y=0.25,
    )
    for i, sim_dir in enumerate(sim_dirs):
        protocol = protocol_names[i]
        print(f"  - Plotting protocol: {protocol}")

        is_last_plot = i == len(sim_dirs) - 1
        current_plotter_kwargs = plotter_kwargs.copy()
        if is_last_plot:
            # Make the last plot wider to accommodate the color bar
            current_plotter_kwargs["window_size"] = [
                plotter_kwargs["window_size"][0] + 75,
                plotter_kwargs["window_size"][1],
            ]

        plotter = pyvista.Plotter(**current_plotter_kwargs)
        plotter.add_text(protocol.capitalize(), font_size=12, position="upper_edge")

        bp_file = sim_dir / "u_growth_pp.bp"
        u_final = get_displacment_at_time(bp_file, mesh, final_times[protocol])

        topo, cells, geom = dolfinx.plot.vtk_mesh(u_final.function_space)
        grid = pyvista.UnstructuredGrid(topo, cells, geom)
        grid.point_data["u"] = u_final.x.array.reshape(grid.n_points, -1)
        grid.cell_data["Displacement (mm)"] = get_disp_magnitude(mesh, u_final) * M2MM
        warped = grid.warp_by_vector("u", factor=warp_scale)

        plotter.add_mesh(
            warped,
            scalars="Displacement (mm)",
            show_edges=False,
            clim=np.array(clim) * M2MM,
            show_scalar_bar=is_last_plot,
            scalar_bar_args=sargs if is_last_plot else None,
        )
        if camera_pos:
            plotter.camera_position = eval(camera_pos)
        else:
            plotter.camera_position = camera_position  # Use the same view as the ungrown plot

        protocol_file = output_file.parent / f"_temp_{i+1}_{protocol}.png"
        plotter.screenshot(protocol_file, scale=4, transparent_background=True)
        temp_files.append(protocol_file)
        plotter.close()

    # --- 4. Stitch images together ---
    print("\n--- Stitching images ---")
    # Ensure all images are loaded as RGB
    images = [imageio.imread(f) for f in temp_files]

    # Combine all images horizontally
    final_image = np.hstack(images)

    # Save the final stitched image
    print(f"Saving final stitched image to {output_file}")
    imageio.imwrite(output_file, final_image)

    # --- 5. Clean up temporary files ---
    print("Cleaning up temporary files...")
    for f in temp_files:
        os.remove(f)

    print("--- Done ---")


if __name__ == "__main__":
    typer.run(main)
