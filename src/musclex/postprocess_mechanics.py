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

from musclex.material import MuscleRohrle
from musclex.utils import get_interpolation_points

# Set pyvista font
pyvista.global_theme.font.family = "arial"

# Matplotlib configuration
from matplotlib import rc

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["font.size"] = 8
plt.rcParams['svg.fonttype'] = 'none'
rc("text", usetex=False)


class PostProcessor:
    def __init__(self, output_dir: Path, conf_file: Path, freq: int = 1):
        print("Initializing Post-Processor...")
        self.sim_dir = output_dir
        self.conf_file = conf_file
        self.results_file = self.sim_dir / "output_pp.bp"
        self.freq = freq
        self.initial_camera_pos = None
        self.activation_levels = np.load(self.sim_dir / "activation_levels.npy")

        self.postprocessed_dir = self.sim_dir.with_name(
            f"{self.sim_dir.name}_postprocessed"
        )
        self.postprocessed_dir.mkdir(exist_ok=True, parents=True)

        # --- Plotting Configuration ---
        self.plot_quantities = {
            "displacement": {
                "title": "",  # Displacement (mm)
                "cmap": "viridis",
                "conversionfactor": 1e3,  # m to mm
            },
            "lambda": {
                "title": "",  # Fiber stretch
                "cmap": "plasma",
                "conversionfactor": 1.0,
            },
            "von_mises": {
                "title": "",  # Von Mises stress (kPa)
                "cmap": "GnBu",
                "conversionfactor": 1e-3,  # Pa to kPa
            },
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
        self.ft = adios4dolfinx.read_meshtags(
            self.results_file, self.domain, meshtag_name="facet_tags"
        )

        print("Reading fiber direction from file...")
        V = dolfinx.fem.functionspace(
            self.domain, ("Lagrange", 2, (self.domain.geometry.dim,))
        )
        self.a0 = dolfinx.fem.Function(V, name="fiberdirection")
        adios4dolfinx.read_function(self.results_file, self.a0, time=0.0)

        print("Re-instantiating material model on loaded mesh...")
        # Use a temporary "decoy" directory to prevent overwriting real results
        decoy_dir = self.sim_dir / "tmp_postprocessing"
        self.material_local = MuscleRohrle(
            self.domain, self.ft, str(self.conf_file), self.a0, decoy_dir
        )

    def _calculate_derived_quantities(self, u, p, activation) -> tuple[dict, dict]:
        """
        Calculates all derived quantities for a time step.

        Returns:
            A tuple containing two dictionaries:
            1. field_quantities: For 3D plotting
            2. scalar_quantities: For line plots
        """
        # --- Kinematics ---
        I = ufl.Identity(self.domain.geometry.dim)
        F = I + ufl.grad(u)
        C = F.T * F
        S = self.material_local.stress_PK2(activation, F, p)
        P = self.material_local.stress_PK1(S, F)
        sigma = self.material_local.stress_Cauchy(P, F)

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

        # Fiber stretch
        lambda_func = dolfinx.fem.Function(DG0)
        I4 = ufl.inner(C * self.a0, self.a0)
        lmbda = ufl.sqrt(I4)
        lambda_expr = dolfinx.fem.Expression(
            lmbda, get_interpolation_points(DG0.element)
        )
        lambda_func.interpolate(lambda_expr)
        field_quantities["lambda"] = lambda_func

        # Von Mises stress
        vm_func = dolfinx.fem.Function(DG0)
        sigma_vm = self.material_local.stress_VM(P, F)
        vm_expr = dolfinx.fem.Expression(
            sigma_vm, get_interpolation_points(DG0.element)
        )
        vm_func.interpolate(vm_expr)
        field_quantities["von_mises"] = vm_func

        # --- 2. Calculate scalar quantities for line plots ---
        force = self.material_local.reaction_force(sigma)
        scalar_quantities = {"reaction_force": force}

        return field_quantities, scalar_quantities

    def _generate_plot_frame(
        self,
        u_data,
        scalar_data,
        scalar_key,
        clim,
        u_space,
        activation_level,
    ):
        """Creates and saves a single PNG frame.

        Args:
            u_data: Displacement data for warping the mesh.
            scalar_data: Scalar data for coloring the mesh.
            scalar_key: Key to identify the scalar quantity (e.g., 'displacement').
            clim: Color limits for the scalar data.
            u_space: Function space for the displacement data.
            activation_level: Current activation level for annotation.
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
        warped = grid.warp_by_vector("u", factor=1.0)

        # Load config for the scalar quantity
        config = self.plot_quantities[scalar_key]

        sargs = dict(
            title=config["title"],
            vertical=True,
            title_font_size=20,
            label_font_size=16,
            shadow=False,
            n_labels=2,  # Number of labels on the colorbar
            fmt="%.1f",
            position_x=0.78,
            position_y=0.3,
            # width=0.08,
            # height=0.4,
            #position_x=0.3,
            #position_y=0.01,
            #width=0.4,
            #height=0.1,
        )

        plotter = pyvista.Plotter(off_screen=True, window_size=[290, 312])
        plotter.add_mesh(
            warped,
            scalars=scalar_key,
            show_edges=False,
            cmap=config["cmap"],
            scalar_bar_args=sargs,
            clim=np.array(clim) * config["conversionfactor"],  # Apply unit conversion
        )

        # Annotate activation level
        #plotter.add_text(
        #    f"Activation level: {activation_level:.2f}",
        #    position="top",
        #    font_size=20,
        #)

        if self.initial_camera_pos is None:
            plotter.view_yz()
            # Save the current camera position
            self.initial_camera_pos = plotter.camera.position
            self.focal_point = plotter.camera.focal_point
            self.up = plotter.camera.up

        plotter.camera.position = self.initial_camera_pos
        plotter.camera.focal_point = self.focal_point
        plotter.camera.up = self.up

        plot_filename = plot_dir / f"act_{activation_level:.3f}.png"
        plotter.screenshot(plot_filename, transparent_background=False, scale=4)
        plotter.close()

    def _plot_reaction_force(self, activation_levels, force_values):
        """Creates and saves a line plot of reaction force vs. activation level."""
        print("\nGenerating force-activation plot...")
        plt.figure(figsize=(2, 1.2))
        plt.plot(activation_levels, force_values, ".-", color="black")
        plt.xlabel(r"Activation level $\alpha$")
        plt.ylabel("Reaction force (N)")

        # Adjust style
        plt.gca().spines["top"].set_visible(False)
        plt.gca().spines["right"].set_visible(False)
        plt.gca().tick_params(top=False, right=False)
        plt.gca().minorticks_off()

        # Save the plot
        filename = self.postprocessed_dir / "force-activation.png"
        plt.savefig(filename, dpi=400, bbox_inches="tight")
        plt.close()
        print(f"  - Plot saved to {filename}")

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
            frames = sorted(plot_dir.glob("act_*.png"))

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
                fps=3,
            ) as writer:
                for frame in frame_data:
                    writer.append_data(frame)

            # --- Create MP4 ---
            mp4_path = video_dir / f"animation_{scalar_key}.mp4"
            print(f"  - Creating MP4 for '{scalar_key}'...")
            with imageio.get_writer(
                mp4_path,
                mode="I",
                fps=3,
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

        u = dolfinx.fem.Function(self.material_local.V, name="u")
        p = dolfinx.fem.Function(self.material_local.Q, name="p")

        for i, activation in enumerate(self.activation_levels):
            print(f"  - Processing frame {i+1}/{len(self.activation_levels)}...")

            try:
                adios4dolfinx.read_function(self.results_file, u, time=activation)
                adios4dolfinx.read_function(self.results_file, p, time=activation)

                field_funcs, scalar_vals = self._calculate_derived_quantities(
                    u, p, activation
                )

                cached_scalar_data.append(scalar_vals)

                frame_data = {"u": u.x.array.copy()}
                for key, func in field_funcs.items():
                    frame_data[key] = func.x.array.copy()

                    min_val, max_val = func.x.array.min(), func.x.array.max()
                    scalar_limits[key][0] = min(scalar_limits[key][0], min_val)
                    scalar_limits[key][1] = max(scalar_limits[key][1], max_val)

                cached_field_data.append(frame_data)
            except:
                break
        epsilon = 1e-10
        for key in scalar_limits.keys():
            if abs(scalar_limits[key][0]) < epsilon:
                scalar_limits[key][0] = 0.0
        print("Data processing complete. Global limits found:")
        for key, limits in scalar_limits.items():
            print(f"  - {key}: [{limits[0]:.4e}, {limits[1]:.4e}]")

        reaction_forces = [d["reaction_force"] for d in cached_scalar_data]
        self._plot_reaction_force(self.activation_levels, reaction_forces)

        # --- 2. Generate 3D plot frames ---
        print("\nGenerating 3D plot frames from cached data...")
        for i in range(0, len(cached_field_data), self.freq):
            frame_data = cached_field_data[i]
            activation_level = self.activation_levels[i]
            print(
                f"  - Plotting frame {i+1}/{len(cached_field_data)} (Activation: {activation_level:.3f})..."
            )
            for key in self.plot_quantities.keys():
                self._generate_plot_frame(
                    u_data=frame_data["u"],
                    scalar_data=frame_data[key]
                    * self.plot_quantities[key]["conversionfactor"],
                    scalar_key=key,
                    clim=scalar_limits[key],
                    u_space=self.material_local.V,
                    activation_level=activation_level,
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
):
    processor = PostProcessor(output_dir, conf_file, freq)
    processor.run()


if __name__ == "__main__":
    app()
