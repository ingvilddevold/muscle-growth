import typer
import numpy as np
import dolfinx
import ufl
import adios4dolfinx
from mpi4py import MPI
import pyvista as pv
from pathlib import Path
from typing_extensions import Annotated
from musclex.utils import get_interpolation_points

# --- Matplotlib Styling ---
from matplotlib import rc
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 8
rc("text", usetex=False)
pv.global_theme.font.family = "arial"


def calculate_J(u: dolfinx.fem.Function) -> dolfinx.fem.Function:
    """Calculates J = det(F) = det(I + grad(u)) on DG0 space."""
    domain = u.function_space.mesh
    I = ufl.Identity(domain.geometry.dim)
    F = I + ufl.grad(u)
    J = ufl.det(F)
    V_J = dolfinx.fem.functionspace(domain, ("DG", 0))
    J_func = dolfinx.fem.Function(V_J, name="J")
    points = get_interpolation_points(V_J.element)
    try:
        J_expr = dolfinx.fem.Expression(J, points, domain.comm)
    except TypeError:
        J_expr = dolfinx.fem.Expression(J, points)
    J_func.interpolate(J_expr)
    return J_func


def calculate_displacement_magnitude(u: dolfinx.fem.Function) -> dolfinx.fem.Function:
    """Calculates ||u|| on DG0 space."""
    domain = u.function_space.mesh
    DG0 = dolfinx.fem.functionspace(domain, ("DG", 0))
    disp_mag_func = dolfinx.fem.Function(DG0, name="DispMag")
    points = get_interpolation_points(DG0.element)
    try:
        disp_expr = dolfinx.fem.Expression(ufl.sqrt(ufl.dot(u, u)), points, domain.comm)
    except TypeError:
        disp_expr = dolfinx.fem.Expression(ufl.sqrt(ufl.dot(u, u)), points)
    disp_mag_func.interpolate(disp_expr)
    return disp_mag_func


@typer.run
def main(
    sim_dirs: Annotated[
        str,
        typer.Option(help="Comma-separated paths to simulation output directories."),
    ],
    labels: Annotated[
        str,
        typer.Option(
            help="Comma-separated labels for each sim_dir (e.g., baseline,var1,...)."
        ),
    ],
    output_file: Annotated[
        Path,
        typer.Option(
            help="Path to save the output grid PNG plot (ignored if --save-individual)."
        ),
    ],
    ode_bp_filename: Annotated[
        str, typer.Option(help="Filename of the ODE history BP file.")
    ],
    warp_scale: Annotated[
        float, typer.Option(help="Scale factor for warping the mesh.")
    ] = 1.0,
    save_individual: Annotated[
        bool,
        typer.Option(
            help="Save each plot individually in its sim_dir instead of a grid."
        ),
    ] = False,
):
    """
    Loads data from multiple simulations.
    - Default: Plots z, J, and ||u|| in a 3x(N+1) grid to 'output_file'.
    - If --save-individual: Saves separate plots for z, J, and ||u||
      into each respective 'sim_dir'.
    """
    pv.set_plot_theme("document")

    sim_dir_paths = [Path(p) for p in sim_dirs.split(",")]
    label_list = labels.split(",")
    n_sims = len(sim_dir_paths)

    all_data = {}
    all_z_values, all_J_values, all_disp_mag_values = [], [], []

    # --- 1. Load data ---
    domain = None
    V_u = None
    base_grid_DG0 = None
    base_grid_P2 = None
    for i, sim_dir in enumerate(sim_dir_paths):
        label = label_list[i]
        print(f"Loading data for: {label} ({sim_dir})")
        data_entry = {}
        # Paths to data files
        bp_file = sim_dir / "u_growth_pp.bp"
        ode_bp_file = sim_dir / ode_bp_filename
        times_file = sim_dir / "output_times.npy"
        ode_times_file = sim_dir / "ode_times.npy"

        # --- Load mesh and function spaces ---
        if domain is None:
            domain = adios4dolfinx.read_mesh(
                bp_file,
                MPI.COMM_WORLD,
                engine="BP4",
                ghost_mode=dolfinx.mesh.GhostMode.none,
            )
            V_u = dolfinx.fem.functionspace(
                domain, ("Lagrange", 2, (domain.geometry.dim,))
            )
            topo_dg0, cells_dg0, geom_dg0 = dolfinx.plot.vtk_mesh(V_u)
            base_grid_DG0 = pv.UnstructuredGrid(topo_dg0, cells_dg0, geom_dg0)
            topo_p2, cells_p2, geom_p2 = dolfinx.plot.vtk_mesh(V_u)
            base_grid_P2 = pv.UnstructuredGrid(topo_p2, cells_p2, geom_p2)

        # --- Load z data ---
        V_plot_DG0 = dolfinx.fem.functionspace(domain, ("DG", 0))
        V_z = V_plot_DG0  # Assuming z is DG0
        z_func = dolfinx.fem.Function(V_z, name="z")

        ode_times = np.load(ode_times_file)
        time_point = float(ode_times[-1])

        try:
            adios4dolfinx.read_function(ode_bp_file, z_func, time=time_point)
            z_cell_data = z_func.x.array.copy()
            if len(z_cell_data) == base_grid_DG0.n_cells:
                data_entry["z_data"] = z_cell_data
                all_z_values.extend(z_cell_data)
            else:
                print(f"Warning: z data size mismatch for {label}")
        except Exception as e:
            print(
                f"    Warning: Could not read z data for {label} at time {time_point}: {e}"
            )

        # --- Load final displacement u ---
        growth_times = np.load(times_file)
        last_time = growth_times[-1]
        u_final = dolfinx.fem.Function(V_u, name="u_growth")
        adios4dolfinx.read_function(bp_file, u_final, time=last_time)

        # Calculate J (local volume change)
        J_func = calculate_J(u_final)
        J_cell_data = J_func.x.array.copy()
        if len(J_cell_data) == base_grid_DG0.n_cells:
            data_entry["J_data"] = J_cell_data
            all_J_values.extend(J_cell_data)

        # Calculate displacement magnitude
        M2MM = 1e3  # Convert from m to mm
        disp_mag_func = calculate_displacement_magnitude(u_final)
        disp_mag_data = disp_mag_func.x.array.copy() * M2MM
        if len(disp_mag_data) == base_grid_DG0.n_cells:
            data_entry["u_mag_data"] = disp_mag_data
            all_disp_mag_values.extend(disp_mag_data)
        data_entry["u_vec_data"] = u_final.x.array.copy()
        all_data[label] = data_entry

    if len(all_data) == 0:
        print("Error: No data loaded.")
        raise typer.Exit(code=1)

    # --- 2. Determine shared color limits ---
    z_clim = np.percentile(all_z_values, [2, 98]) if all_z_values else [0, 1]
    J_clim = np.percentile(all_J_values, [2, 98]) if all_J_values else [0, 1]
    disp_clim = np.percentile(all_disp_mag_values, [2, 98]) if all_disp_mag_values else [0, 1]
    for clim in [z_clim, J_clim, disp_clim]:
        if clim[0] == clim[1]:
            clim[1] += 1e-9

    # --- 3. Define Plot Settings ---
    quantities = ["z", "J", "u_mag"]
    titles = {
        "z": "", #"Final N",
        "J": "", #"Final J",
        "u_mag": "", #"Final Disp. Mag.",
    }
    cmaps = {"z": "Blues", "J": "Greens", "u_mag": "viridis"}
    clims = {"z": z_clim, "J": J_clim, "u_mag": disp_clim}
    fmts = {"z": "%.2f", "J": "%.3f", "u_mag": "%.2f"}
    data_keys = {"z": "z_data", "J": "J_data", "u_mag": "u_mag_data"}

    # --- 4. Get Camera Position ---
    ref_plotter = pv.Plotter(off_screen=True)
    ref_plotter.add_mesh(base_grid_DG0)
    ref_plotter.view_yz()
    ref_plotter.camera.zoom(1.3)
    camera_pos = ref_plotter.camera_position
    ref_plotter.close()

    # --- 5. Plotting ---
    if save_individual:
        # --- 5a. Save Individual Plots ---
        print("Saving individual plots...")

        sbar_args_common = {
            "vertical": True,
            "n_labels": 2,
            "title_font_size": 20,
            "label_font_size": 16,
            "position_x": 0.72,
            "position_y": 0.3,
            "width": 0.1,
            "height": 0.4,
            "shadow": False,
        }

        for i, label in enumerate(label_list):
            sim_dir = sim_dir_paths[i]
            print(f"  Processing: {label} (in {sim_dir})")

            for qty in quantities:
                plotter = pv.Plotter(
                    window_size=[290, 312],
                    off_screen=True,
                    border=False,
                )

                if label not in all_data or data_keys[qty] not in all_data[label]:
                    plotter.add_text("Data Missing", font_size=10)
                else:
                    # --- Get data and warp mesh ---
                    grid_dg0 = base_grid_DG0.copy()
                    grid_dg0.cell_data[qty] = all_data[label][data_keys[qty]]

                    grid_p2 = base_grid_P2.copy()
                    grid_p2.point_data["u"] = all_data[label][
                        "u_vec_data"
                    ].reshape(grid_p2.n_points, -1)
                    warped_grid = grid_p2.warp_by_vector("u", factor=warp_scale)
                    warped_grid.cell_data[qty] = all_data[label][data_keys[qty]]
                    grid_to_plot = warped_grid
                    scalars_to_plot = qty
                    # --- End data/warp ---

                    # Add mesh with its own scalar bar
                    plotter.add_mesh(
                        grid_to_plot,
                        scalars=scalars_to_plot,
                        cmap=cmaps[qty],
                        clim=clims[qty],
                        show_edges=False,
                        scalar_bar_args={
                            "title": titles[qty],
                            "fmt": fmts[qty],
                            **sbar_args_common,
                        },
                        show_scalar_bar=True,
                    )
                    plotter.camera_position = camera_pos
                    # Add title for the individual plot
                    #plotter.add_text(
                    #    f"{label.capitalize()}: {titles[qty]}",
                    #    font="arial",
                    #    position="upper_center",
                    #    font_size=12,
                    #)

                # --- Save the individual plot ---
                output_path = sim_dir.parent / (sim_dir.stem + "_postprocessed") / f"{qty}_final_plot.png"

                output_path.parent.mkdir(parents=True, exist_ok=True)
                plotter.screenshot(output_path)
                plotter.close()
                print(f"    Saved: {output_path}")

        print(f"\nSuccessfully saved all individual plots.")

    else:
        # --- 5b. Create the Grid Plot (Original Logic) ---
        print(f"Saving grid plot to {output_file}...")
        n_cols_total = n_sims + 1  # Add one column for scalar bars
        plotter = pv.Plotter(
            shape=(3, n_cols_total),
            window_size=[150 * n_cols_total, 800],
            off_screen=True,
            border=False,
        )

        mappers = {"z": None, "J": None, "u_mag": None}

        for row, qty in enumerate(quantities):
            for col, label in enumerate(label_list):
                plotter.subplot(row, col)

                if label not in all_data or data_keys[qty] not in all_data[label]:
                    plotter.add_text("Data Missing", font_size=10)
                    continue

                # --- Get data and warp mesh ---
                grid_dg0 = base_grid_DG0.copy()
                grid_dg0.cell_data[qty] = all_data[label][data_keys[qty]]

                grid_p2 = base_grid_P2.copy()
                grid_p2.point_data["u"] = all_data[label]["u_vec_data"].reshape(
                    grid_p2.n_points, -1
                )
                warped_grid = grid_p2.warp_by_vector("u", factor=warp_scale)
                warped_grid.cell_data[qty] = all_data[label][data_keys[qty]]
                grid_to_plot = warped_grid
                scalars_to_plot = qty
                # --- End data/warp ---

                actor = plotter.add_mesh(
                    grid_to_plot,
                    scalars=scalars_to_plot,
                    cmap=cmaps[qty],
                    clim=clims[qty],
                    show_edges=False,
                    scalar_bar_args=None,
                    show_scalar_bar=False,
                )

                if (
                    col == 0
                    and mappers[qty] is None
                    and actor
                    and hasattr(actor, "mapper")
                ):
                    mappers[qty] = actor.mapper

                # Add labels
                #if row == 0:
                #    plotter.add_text(
                #        label.capitalize(),
                #        font="arial",
                #        position="upper_center",
                #        font_size=9,
                #    )
                #if col == 0:
                #    plotter.add_text(
                #        titles[qty],
                #        font="arial",
                #        position="center_left",
                #        font_size=9,
                #        orientation=90,
                #    )

                plotter.camera_position = camera_pos

        # --- Add shared scalar bars ---
        sbar_col_index = n_sims
        sbar_args_common = {
            "vertical": True,
            "n_labels": 4,
            "title_font_size": 9,
            "label_font_size": 8,
            "position_x": 0.4,
            "position_y": 0.2,
        }

        for row, qty in enumerate(quantities):
            plotter.subplot(row, sbar_col_index)
            if mappers[qty]:
                plotter.add_scalar_bar(
                    title=titles[qty],
                    mapper=mappers[qty],
                    fmt=fmts[qty],
                    **sbar_args_common,
                )
            else:
                plotter.add_text(f"No data for\n{titles[qty]}", font_size=9, position="center")


        # --- 6. Save the plot ---
        output_file.parent.mkdir(parents=True, exist_ok=True)
        plotter.screenshot(output_file)
        plotter.close()

        print(f"Successfully saved 3D grid plot to {output_file}")