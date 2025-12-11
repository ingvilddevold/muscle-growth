import dolfinx
import ufl
import adios4dolfinx
import numpy as np
import pyvista
from pathlib import Path
from mpi4py import MPI
from matplotlib import pyplot as plt
import imageio
import typer
from musclex.utils import get_interpolation_points


# Set pyvista font
pyvista.global_theme.font.family = "arial"

# Matplotlib configuration
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
rc("text", usetex=False)

CAMERA_POS = {
    "idealized-fusiform": [
        (0.5774328905527678, 0.0, 0.10000000000000002),
        (-1.734723475976807e-18, 0.0, 0.10000000000000002),
        (0.0, 0.0, 1.0),
    ],
    "realistic": [
        (1.0214701186665467, 0.7657064006084325, 1.0527990992216494),
        (0.6389845141892749, 0.3832207961311609, 0.6703134947443776),
        (0.0, 0.0, 1.0),
    ],
}


class PostProcessor:
    def __init__(
        self,
        output_dir: Path,
        conf_file: Path,
        freq: int = 1,
        warp_scale: float = 1.0,
        camera_pos: str = None,
    ):
        print("Initializing Post-Processor...")
        self.sim_dir = output_dir
        self.conf_file = conf_file
        self.results_file = self.sim_dir / "u_growth_pp.bp"
        self.freq = freq
        self.growth_times = np.load(self.sim_dir / "output_times.npy")
        self.warp_scale = warp_scale
        self.camera_pos = CAMERA_POS.get(camera_pos)

        self.postprocessed_dir = self.sim_dir.with_name(
            f"{self.sim_dir.name}_postprocessed"
        )
        self.postprocessed_dir.mkdir(exist_ok=True, parents=True)

        # --- Plotting Configuration ---
        self.plot_quantities = {
            "displacement": {"title": "Displacement (m)", "cmap": "viridis"},
        }

        # --- Setup domain and model from files ---
        self._setup_domain_and_model()

    def _setup_domain_and_model(self):
        """Reads mesh and fiber data, then re-instantiates the material model
        to get constants and methods defined on same mesh."""
        print("Reading mesh and facet tags from file...")
        self.domain = adios4dolfinx.read_mesh(
            self.results_file,
            MPI.COMM_WORLD,
            engine="BP4",
            ghost_mode=dolfinx.mesh.GhostMode.none,
        )
        self.V = dolfinx.fem.functionspace(
            self.domain, ("Lagrange", 2, (self.domain.geometry.dim,))
        )

    def _calculate_derived_quantities(self, u) -> tuple[dict, dict]:
        """
        Calculates all derived quantities for a time step.

        Returns:
            A tuple containing two dictionaries:
            1. field_quantities: For 3D plotting
            2. scalar_quantities: For line plots
        """

        # --- 1. Calculate field quantities for 3D plots ---
        field_quantities = {}

        # Displacement
        DG0 = dolfinx.fem.functionspace(self.domain, ("DG", 0))
        disp_mag_func = dolfinx.fem.Function(DG0)
        disp_expr = dolfinx.fem.Expression(
            ufl.sqrt(ufl.dot(u, u)), get_interpolation_points(DG0.element)
        )
        disp_mag_func.interpolate(disp_expr)
        field_quantities["displacement"] = disp_mag_func

        # --- 2. Calculate scalar quantities for line plots ---
        scalar_quantities = {}

        return field_quantities, scalar_quantities

    def _generate_plot_frame(
        self,
        u_data,
        scalar_data,
        scalar_key,
        clim,
        u_space,
        time,
        camera_position=None,
    ):
        """Creates and saves a single PNG frame.

        Args:
            u_data: Displacement data for warping the mesh.
            scalar_data: Scalar data for coloring the mesh.
            scalar_key: Key to identify the scalar quantity (e.g., 'displacement').
            clim: Color limits for the scalar data.
            u_space: Function space for the displacement data.
            time: Current time for annotation.
            camera_position: Optional camera position to apply to the plotter.
        
        Returns:
            The camera position used by the plotter if no position was provided.
        """
        plot_dir = self.postprocessed_dir / f"plots_{scalar_key}"
        plot_dir.mkdir(exist_ok=True, parents=True)

        # Construct pyvista grid
        topology, cell_types, geometry = dolfinx.plot.vtk_mesh(u_space)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Load the data into the grid
        grid.point_data["u"] = u_data.reshape(grid.n_points, -1)
        grid.cell_data[scalar_key] = scalar_data

        # Warp the mesh by the displacement vector
        warped = grid.warp_by_vector("u", factor=self.warp_scale)

        # Load config for the scalar quantity
        config = self.plot_quantities[scalar_key]

        sargs = dict(
            title=config["title"],
            vertical=False,
            title_font_size=16,
            label_font_size=14,
            shadow=False,
            n_labels=2,  # Number of labels on the colorbar
            fmt="%.1g",
            # position_x=0.68,
            # position_y=0.3,
            # width=0.08,
            # height=0.4,
            position_x=0.2,
            position_y=0.01,
            width=0.6,
            height=0.1,
        )

        plotter = pyvista.Plotter(off_screen=True, window_size=[366, 512])
        plotter.add_mesh(
            warped,
            scalars=scalar_key,
            show_edges=False,
            cmap=config["cmap"],
            scalar_bar_args=sargs,
            clim=clim,
        )

        # Annotate time
        plotter.add_text(
            f"Time: {time:.2f}",
            position="top",
            font_size=20,
        )

        if camera_position:
            plotter.camera_position = camera_position
        else:
            plotter.view_yz()
            # Reduce padding
            plotter.camera.zoom(1.25)

        plot_filename = plot_dir / f"time_{time:06.1f}.png"
        plotter.screenshot(plot_filename, transparent_background=True, scale=5)
        
        # If no camera position was provided, return the one we generated
        if camera_position is None:
            final_camera_pos = plotter.camera_position
            plotter.close()
            return final_camera_pos
        
        plotter.close()

    def _create_media_files(self):
        """Creates both GIF and MP4 files from the generated PNG frames."""
        print("\nCreating final media files (GIF and MP4)...")

        # Ensure the video directory exists
        video_dir = self.postprocessed_dir / "video"
        video_dir.mkdir(exist_ok=True)

        # Iterate over each scalar key to create animations
        for scalar_key in self.plot_quantities.keys():
            plot_dir = self.postprocessed_dir / f"plots_{scalar_key}"
            # Generate a list of paths to PNG frames
            frames = sorted(plot_dir.glob("time_*.png"))

            if not frames:
                print(f"  - No frames found for '{scalar_key}', skipping.")
                continue

            # Read all frames into a list
            frame_data = [imageio.imread(frame) for frame in frames]

            # --- Create GIF ---
            gif_path = video_dir / f"animation_{scalar_key}.gif"
            print(f"  - Creating GIF for '{scalar_key}'...")
            with imageio.get_writer(
                gif_path,
                mode="I",
                loop=0,
                fps=8,
            ) as writer:
                for frame in frame_data:
                    rgb_frame = frame[..., :3]
                    writer.append_data(rgb_frame)

            # --- Create MP4 ---
            mp4_path = video_dir / f"animation_{scalar_key}.mp4"
            print(f"  - Creating MP4 for '{scalar_key}'...")
            with imageio.get_writer(
                mp4_path,
                mode="I",
                fps=8,
            ) as writer:
                for frame in frame_data:
                    writer.append_data(frame)

    def run(self):
        """Runs the entire post-processing pipeline:"""

        # --- 1. Read all data to determine clims ---
        print("Starting data processing...")

        cached_field_data = []
        cached_scalar_data = []
        scalar_limits = {
            key: [float("inf"), float("-inf")] for key in self.plot_quantities
        }

        u = dolfinx.fem.Function(self.V, name="u_growth")

        # pyvista.start_xvfb()

        for i, time in enumerate(self.growth_times):
            print(f"  - Processing frame {i+1}/{len(self.growth_times)}...")

            adios4dolfinx.read_function(self.results_file, u, time=time)

            field_funcs, scalar_vals = self._calculate_derived_quantities(
                u,
            )

            cached_scalar_data.append(scalar_vals)

            frame_data = {"u": u.x.array.copy()}
            for key, func in field_funcs.items():
                frame_data[key] = func.x.array.copy()

                min_val, max_val = func.x.array.min(), func.x.array.max()
                scalar_limits[key][0] = min(scalar_limits[key][0], min_val)
                scalar_limits[key][1] = max(scalar_limits[key][1], max_val)

            cached_field_data.append(frame_data)

        epsilon = 1e-10
        for key in scalar_limits.keys():
            if abs(scalar_limits[key][0]) < epsilon:
                scalar_limits[key][0] = 0.0
        print("Data processing complete. Global limits found:")
        for key, limits in scalar_limits.items():
            print(f"  - {key}: [{limits[0]:.4e}, {limits[1]:.4e}]")

        # --- 2. Generate 3D plot frames ---
        print("\nGenerating 3D plot frames from cached data...")

        # --- Plot first frame and capture camera position if needed ---
        if len(cached_field_data) > 0:
            i = 0
            frame_data = cached_field_data[i]
            time = self.growth_times[i]
            print(
                f"  - Plotting frame {i+1}/{len(cached_field_data)} (Time: {time:.3f})..."
            )
            for key in self.plot_quantities.keys():
                plot_dir = self.postprocessed_dir / f"plots_{key}"
                plot_dir.mkdir(exist_ok=True, parents=True)

                # If camera position is not set, capture it from the first frame
                if self.camera_pos is None:
                    cam_pos = self._generate_plot_frame(
                        u_data=frame_data["u"],
                        scalar_data=frame_data[key],
                        scalar_key=key,
                        clim=scalar_limits[key],
                        u_space=self.V,
                        time=time,
                        camera_position=None,
                    )
                    if cam_pos is not None:
                        print(f"  - Camera position captured: {cam_pos}")
                        self.camera_pos = cam_pos
                        # Re-plot the first frame with the captured camera position
                        print("  - Re-plotting first frame for consistency...")
                        self._generate_plot_frame(
                            u_data=frame_data["u"],
                            scalar_data=frame_data[key],
                            scalar_key=key,
                            clim=scalar_limits[key],
                            u_space=self.V,
                            time=time,
                            camera_position=self.camera_pos,
                        )

                else:
                    self._generate_plot_frame(
                        u_data=frame_data["u"],
                        scalar_data=frame_data[key],
                        scalar_key=key,
                        clim=scalar_limits[key],
                        u_space=self.V,
                        time=time,
                        camera_position=self.camera_pos,
                    )

        # --- Plot remaining frames ---
        for i in range(self.freq, len(cached_field_data), self.freq):
            frame_data = cached_field_data[i]
            time = self.growth_times[i]
            print(
                f"  - Plotting frame {i+1}/{len(cached_field_data)} (Time: {time:.3f})..."
            )
            for key in self.plot_quantities.keys():
                plot_dir = self.postprocessed_dir / f"plots_{key}"
                plot_dir.mkdir(exist_ok=True, parents=True)

                self._generate_plot_frame(
                    u_data=frame_data["u"],
                    scalar_data=frame_data[key],
                    scalar_key=key,
                    clim=scalar_limits[key],
                    u_space=self.V,
                    time=time,
                    camera_position=self.camera_pos,
                )

        # Include the final time point if it wasn't plotted
        last_idx = len(cached_field_data) - 1
        if last_idx > 0 and last_idx % self.freq != 0:
            i = last_idx
            frame_data = cached_field_data[i]
            time = self.growth_times[i]
            print(
                f"  - Plotting final frame {i+1}/{len(cached_field_data)} (Time: {time:.3f})..."
            )
            for key in self.plot_quantities.keys():
                # Set up output directory
                plot_dir = self.postprocessed_dir / f"plots_{key}"
                plot_dir.mkdir(exist_ok=True, parents=True)

                self._generate_plot_frame(
                    u_data=frame_data["u"],
                    scalar_data=frame_data[key],
                    scalar_key=key,
                    clim=scalar_limits[key],
                    u_space=self.V,
                    time=time,
                    camera_position=self.camera_pos,
                )

        # --- 3. Create media files ---
        self._create_media_files()

        print("\nPost-processing complete.")


app = typer.Typer()


@app.command()
def postprocess_from_file(
    output_dir: Path = typer.Argument(
        ...,
        help="Path to simulation results directory.",
        exists=True,
        file_okay=False,
        dir_okay=True,
        resolve_path=True,
    ),
    conf_file: Path = typer.Argument(
        ...,
        help="Path to material configuration file (.yml).",
        exists=True,
        file_okay=True,
        dir_okay=False,
        resolve_path=True,
    ),
    freq: int = typer.Option(
        1,
        "--freq",
        help="Frequency of data saving in the simulation (default: 1).",
    ),
    warp_scale: float = typer.Option(
        1.0,
        "--warp-scale",
        help="Scaling factor for displacement visualization (default: 1.0).",
    ),
    camera_pos: str = typer.Option(
        None,
        "--camerapos",
        help="Camera position for 3D plots.",
    ),
):
    processor = PostProcessor(output_dir, conf_file, freq, warp_scale, camera_pos)
    processor.run()


if __name__ == "__main__":
    app()
