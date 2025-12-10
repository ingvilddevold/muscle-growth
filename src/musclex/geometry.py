import dolfinx
from mpi4py import MPI
from petsc4py import PETSc
import ufl
import adios4dolfinx
#import pyvista
import dolfinx.plot as plot
import numpy as np
from pathlib import Path
import typing
from typing import NamedTuple
from musclex.utils import get_interpolation_points


mesh_folder = Path(__file__).parents[2] / "meshes"


class Marker(NamedTuple):
    """
    Marker for facet tags.

    Args:
        name (str): The name of the marker.
        marker (int): The integer value representing the marker.
        dim (int): The dimension associated with the marker.
        locator (Callable[[np.typing.NDArray[np.float64]], bool]): A callable that takes a numpy array of floats and returns a boolean indicating the location.
    """

    name: str
    marker: int
    dim: int
    locator: typing.Callable[[np.typing.NDArray[np.float64]], bool]


class Geometry:
    domain: dolfinx.mesh.Mesh
    fibers: ufl.core.expr.Expr
    boundaries: typing.Sequence[Marker]
    ft: dolfinx.mesh.MeshTags
    name: str

    def __init__(
        self,
        domain: dolfinx.mesh.Mesh,
        fibers: ufl.core.expr.Expr,
    ):
        self.domain = domain
        self.fibers = fibers

    @property
    def gdim(self) -> int:
        """Return the geometric dimension of the mesh"""
        return self.domain.geometry.dim

    @property
    def tdim(self) -> int:
        """Return the topological dimension of the mesh"""
        return self.domain.topology.dim

    @property
    def fdim(self) -> int:
        """Return the dimension of the facets"""
        return self.domain.topology.dim - 1

    @property
    def volume(self) -> float:
        """Return the total volume of the mesh"""
        return dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(
                dolfinx.fem.Constant(self.domain, PETSc.ScalarType(1))
                * ufl.dx(self.domain)
            )
        )

    def compute_csa(self, u=None):
        """Compute cross-sectional area of the muscle."""
        # Assumes center cross-section is tagged with 3
        # Using Nanson's formula,
        #   ds = J ||F^-T N|| dS

        dS = ufl.Measure("dS", self.domain, subdomain_data=self.ft)

        if u:  # compute deformed CSA
            F = ufl.Identity(self.gdim) + ufl.grad(u)
            J = ufl.det(F)  # volume change
            N = ufl.FacetNormal(self.domain)
            f = ufl.inv(F.T) * N  # argument to norm
            norm = ufl.sqrt(ufl.dot(f, f))
            csa_local = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(J("+") * norm("+") * dS(3))
            )
        else:  # compute undeformed CSA
            csa_local = dolfinx.fem.assemble_scalar(dolfinx.fem.form(1 * dS(3)))

        return self.domain.comm.allreduce(csa_local, op=MPI.SUM)

    def plot(
        self,
        mode="all",  # "mesh", "facet_tags", "fibers", or "all"
        fiber_opacity=0.5,
        glyph_scale_factor=20,
        padding=0.05,
        filename=None,
        fiber_streamlines=False,
        glyph_subsampling=1,
    ):
        """
        Plot the mesh, facet tags, fibers, or all side by side.

        Args:
            mode: "mesh", "facet_tags", "fibers", or "all"
            fiber_opacity: Opacity for the mesh in the fibers plot
            glyph_scale_factor: Scale for fiber glyphs (higher = longer arrows)
            padding: Padding around the view
            filename: If provided, saves the plot to this file instead of showing it.
            fiber_streamlines: If True, uses streamlines to visualize the fiber field.
            glyph_subsampling: Subsampling factor for glyphs (e.g., 10 means plot every 10th glyph).
        """
        from matplotlib.colors import ListedColormap
        import pyvista
        #pyvista.start_xvfb()
        pyvista.global_theme.camera.viewup = [0, 0, 1]

        tdim = self.domain.topology.dim
        fdim = tdim - 1

        # Mesh
        topology, cell_types, geometry = plot.vtk_mesh(self.domain, tdim)
        grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)

        # Facet tags
        self.domain.topology.create_connectivity(fdim, tdim)
        facet_topology, facet_cell_types, facet_geometry = plot.vtk_mesh(
            self.domain, fdim
        )
        facet_grid = pyvista.UnstructuredGrid(
            facet_topology, facet_cell_types, facet_geometry
        )
        facet_values = np.zeros(facet_grid.n_cells, dtype=int)
        if hasattr(self, "ft"):
            tag_indices = self.ft.indices
            tag_values = self.ft.values
            for i, idx in enumerate(tag_indices):
                facet_values[idx] = tag_values[i]
        cmap = ListedColormap(
            [
                (0, 0, 0, 0),  # transparent for untagged
                (52 / 255, 120 / 255, 0 / 255, 1.0),  # Green for marker 1
                (52 / 255, 120 / 255, 0 / 255, 1.0),  # Green for marker 2
                (0.0, 0.0, 0.0, 0.0),  # transparent for marker 3 (slip)
            ]
        )

        # Fibers
        fibers_vis = self.fiber_function(deg=1)
        fiber_topology, fiber_cell_types, fiber_geometry = plot.vtk_mesh(
            fibers_vis.function_space
        )
        fiber_grid = pyvista.UnstructuredGrid(
            fiber_topology, fiber_cell_types, fiber_geometry
        )
        fiber_grid.point_data["fibers"] = fibers_vis.x.array.reshape(
            fiber_geometry.shape[0], tdim
        )

        # Determine points for glyphing (subsample if requested)
        if glyph_subsampling > 1 and not fiber_streamlines:
            points_to_glyph = pyvista.PolyData(fiber_grid.points[::glyph_subsampling])
            points_to_glyph["fibers"] = fiber_grid.point_data["fibers"][::glyph_subsampling]
        else:
            points_to_glyph = fiber_grid
        
        domain_size = np.linalg.norm(
            np.array(fiber_grid.bounds[1::2]) - np.array(fiber_grid.bounds[::2])
        )
        glyph_scale = domain_size / glyph_scale_factor
        glyphs = points_to_glyph.glyph(
            orient="fibers", factor=glyph_scale, geom=pyvista.Arrow()
        )

        # Determine subplots
        if mode == "all":
            plotter = pyvista.Plotter(
                shape=(1, 3),
                window_size=[600, 500],
                off_screen=pyvista.OFF_SCREEN,
                border=False,
            )
            plot_modes = ["mesh", "facet_tags", "fibers"]
        else:
            plotter = pyvista.Plotter(
                shape=(1, 1),
                window_size=[250, 500],
                off_screen=pyvista.OFF_SCREEN,
                border=False,
            )
            plot_modes = [mode]

        for i, m in enumerate(plot_modes):
            plotter.subplot(0, i)
            light = pyvista.Light(light_type="headlight", intensity=0.1)
            plotter.add_light(light)

        for i, m in enumerate(plot_modes):
            plotter.subplot(0, i)
            if m == "mesh":
                plotter.add_mesh(
                    grid,
                    style="surface",
                    color="#f8f8f2",
                    show_edges=True,
                    edge_color="black",
                    show_scalar_bar=False,
                )
            elif m == "facet_tags":
                plotter.add_mesh(
                    grid,
                    style="surface",
                    color="lightgray",
                    opacity=0.2,
                    show_edges=False,
                    show_scalar_bar=False,
                )
                plotter.add_mesh(
                    facet_grid,
                    scalars=facet_values,
                    show_edges=True,
                    cmap=cmap,
                    opacity=1.0, #0.7,
                    clim=[0, 3],
                    show_scalar_bar=False,
                )
            elif m == "fibers":
                plotter.add_mesh(
                    fiber_grid,
                    style="surface",
                    color="gainsboro",
                    opacity=fiber_opacity,
                    show_edges=True,
                    edge_color="gray",
                    show_scalar_bar=False,
                )
                if fiber_streamlines:
                    # Use streamlines to show the "flow" of the vector field
                    # Streamlines are seeded from a sphere placed at the mesh's center
                    streamlines = fiber_grid.streamlines(
                        vectors="fibers",
                        n_points=30,
                        source_center=grid.center,
                        source_radius=grid.bounds[1],
                    )
                    plotter.add_mesh(
                        streamlines,
                        name="fiber_streamlines",
                        color="darkorange",
                        line_width=3,
                    )
                else:

                    plotter.add_mesh(
                        glyphs,
                        name="fiber_glyphs",
                        color=(52 / 255, 120 / 255, 0 / 255),  # "darkorange",
                        show_scalar_bar=False,
                    )
            else:
                raise ValueError(f"Unknown plot mode: {m}")

            # Reduce surrounding space
            #plotter.view_yz()
            #plotter.camera.tight(view="yz", padding=padding, adjust_render_window=False)

            # Adjust camera angle to see planes in the xy-plane
            #plotter.camera.elevation = 30
            plotter.view_yz()
            plotter.camera.zoom(1.5)

        if pyvista.OFF_SCREEN:
            plotter.screenshot(filename, scale=5)
        else:
            plotter.show()
        return plotter

    def fiber_function(self, deg=2):
        """Create a dolfinx Function for a fiber field specified as a vector expression"""
        # Check if self.fibers is already a dolfinx.fem.Function
        # of the correct degree
        if (
            isinstance(self.fibers, dolfinx.fem.Function)
            and self.fibers.function_space.ufl_element().degree == deg
        ):
            print("Using existing fiber function.")
            return self.fibers
        else:
            # Create a vector function space for the fiber directions
            V = dolfinx.fem.functionspace(
                self.domain, ("Lagrange", deg, (self.domain.geometry.dim,))
            )
            fibers = dolfinx.fem.Function(V, name="fiberdirection")
            # Interpolate the fiber expression to the function space
            fibers_expr = dolfinx.fem.Expression(
                self.fibers, get_interpolation_points(V.element), self.domain.comm
            )
            fibers.interpolate(fibers_expr)
            # fibers = musclex.utils.project(self.fibers, V)
        return fibers

    def write_xdmf(self, filename):
        """Write the mesh and facet tags to an XDMF file"""
        with dolfinx.io.XDMFFile(MPI.COMM_WORLD, filename, "w") as f:
            f.write_mesh(self.domain)
            self.domain.topology.create_connectivity(
                self.domain.topology.dim - 1, self.domain.topology.dim
            )
            f.write_meshtags(self.ft, self.domain.geometry)

    def write_bp(self, filename):
        """Write the mesh and fiber orientations to a .bp file"""
        M = self.fiber_function()
        with dolfinx.io.VTXWriter(self.domain.comm, filename, [M], engine="BP4") as vtx:
            vtx.write(0)

    def construct_facet_tags(self):
        """Construct facet tags from markers"""
        facet_indices, facet_markers = [], []

        # Iterate markers
        for name, marker, dim, locator in self.boundaries:
            # Find entities in given dimension that satisfy the locator function
            facets = dolfinx.mesh.locate_entities(self.domain, dim, locator)
            # Append the indices of those entities
            facet_indices.append(facets)
            # Append the marker value for those entities
            facet_markers.append(np.full_like(facets, marker))

        # Concatenate the indices and markers
        self._facet_indices = np.hstack(facet_indices)
        self._facet_markers = np.hstack(facet_markers)
        self._sorted_facets = np.argsort(self._facet_indices)

        # Sort the entity indices and marker values
        entities = self._facet_indices[self._sorted_facets]
        values = self._facet_markers[self._sorted_facets]

        # Construct mesh tags object
        self.ft = dolfinx.mesh.meshtags(
            self.domain,
            self.fdim,
            entities,
            values,
        )

    def info(self):
        """Print information about the geometry"""
        print("\n--- Geometry info ---")
        tdim = self.domain.topology.dim
        print(f"Geometry type: {self.name}")
        print(f"Mesh dimension: {self.gdim}")
        num_cells = self.domain.topology.index_map(tdim).size_local
        print(f"Number of cells: {num_cells}")
        h = dolfinx.cpp.mesh.h(self.domain._cpp_object, tdim, np.arange(num_cells))
        print(f"Mesh size h_max: {h.max():.2e}, h_min: {h.min():.2e}")
        print(f"Volume: {self.volume:.2e} m^3 \n")


class Cuboid(Geometry):
    """Cuboidal geometry with fiber direction in z-direction"""

    def __init__(self, Lx=1, Ly=1, Lz=5, Nx=8, Ny=8, Nz=40):
        """Cuboidal geometry of size Lx * Ly * Lz with Nx / Ny / Nz cells in each direction"""

        # Create cuboid mesh
        domain = dolfinx.mesh.create_box(
            MPI.COMM_WORLD,
            [(0.0, 0.0, 0.0), (Lx, Ly, Lz)],
            (Nx, Ny, Nz),
            dolfinx.mesh.CellType.tetrahedron,
        )
        self.name = "cuboid"

        # Let the fiber direction M be unit vector in z direction
        M = [0] * domain.geometry.dim
        M[2] = 1.0
        M = ufl.as_vector(M)

        # Call the super class constructor
        super().__init__(domain, M)

        # Define locator functions for the ends of the cuboid
        def bottom(x):
            return np.isclose(x[2], 0.0)

        def top(x):
            return np.isclose(x[2], Lz)

        # Define locator function for the middle of the cuboid (used for CSA)
        dz = Lz / Nz

        def middle(x):
            return np.isclose(x[2], Lz / 2 - 1e-3, atol=dz / 2)

        # Define markers for the boundaries
        self.boundaries = [
            Marker("bottom", 1, 2, bottom),
            Marker("top", 2, 2, top),
            Marker("middle", 3, 2, middle),
        ]
        # Create facet tags based on markers
        self.construct_facet_tags()


class UnitCube(Geometry):
    """Unit cube geometry with fiber direction in z-direction"""

    def __init__(
        self,
        Nx: float = 10,
        Ny: float = 10,
        Nz: float = 10,
    ):
        """Unit cube geometry with Nx / Ny / Nz cells in each direction"""
        # Create unit cube mesh
        domain = dolfinx.mesh.create_unit_cube(MPI.COMM_WORLD, Nx, Ny, Nz)
        self.name = "unit_cube"

        # Let fiber direction M be unit vector in z direction
        M = [0] * domain.geometry.dim
        M[2] = 1.0
        M = ufl.as_vector(M)

        # Call the super class constructor
        super().__init__(domain, M)

        # Define locator functions for the ends of the cube
        def bottom(x):
            return np.isclose(x[2], 0.0)

        def top(x):
            return np.isclose(x[2], 1)

        # Define locator function for the middle of the cube (used for CSA)
        def middle(x):
            return np.isclose(x[2], 1 / 2 - 1e-3, atol=0.5 / Nz)

        # Define markers for the boundaries
        self.boundaries = [
            Marker("bottom", 1, 2, bottom),
            Marker("top", 2, 2, top),
            Marker("middle", 3, 2, middle),
        ]
        # Create facetg tags based on markers
        self.construct_facet_tags()


class RealisticGeometry(Geometry):
    """Realistic muscle geometry with pre-computed fiber orientations"""

    def __init__(self, meshfile, fibersfile=None, comm=MPI.COMM_WORLD):

        # Read muscle mesh and facet tags from file
        with dolfinx.io.XDMFFile(comm, meshfile, "r") as f:
            domain = f.read_mesh(name="Grid")
            domain.topology.create_connectivity(
                domain.topology.dim - 1, domain.topology.dim
            )
            self.ft = f.read_meshtags(domain, name="ft_small")
        self.name = "realistic"

        # convert from mm to m
        domain.geometry.x[:] = domain.geometry.x * 1e-3

        # Read fiber orientations from file
        # Note that V_fibers should match the space used when computing the fiber orientation
        if fibersfile:
            V_fibers = dolfinx.fem.functionspace(
                domain,
                (
                    "Lagrange",
                    2,
                    (domain.geometry.dim,),
                ),
            )
            fibers = dolfinx.fem.Function(V_fibers, name="fiberdirection")
            adios4dolfinx.read_function(fibersfile, fibers)
        else:
            print("No fiber orientation file provided. Using z direction.")
            M = [0.0] * domain.geometry.dim
            M[2] = 1.0
            M = ufl.as_vector(M)
            fibers = M

        # Call the super class constructor
        super().__init__(domain, fibers)

        # Initialize attributes for CSA computation
        self.cutting_surface = None
        self.integration_measure = None
        self.intersecting_cells = None
        self.plane_normal = None

    def setup_csa_surface(self, method="plane", resolution=50, thickness=1e-3):
        """
        Performs the one-time setup for CSA computation by creating a cutting surface,
        finding the intersection, and defining the integration measure.

        Args:
            method (str, optional): "plane" or "box". Defaults to "plane".
            resolution (int, optional): Resolution for the plane or box cross-section. Defaults to 50.
            thickness (float, optional): Thickness for the box method. Defaults to 1e-4.
        """
        from musclex import intersection
        
        muscle_mesh = self.domain

        if method == "plane":
            plane_mesh, normal = intersection.create_midpoint_plane(
                muscle_mesh, resolution=resolution
            )
            self.cutting_surface = plane_mesh
            self.plane_normal = normal
            colliding_muscle_tags, intersecting_cell_tags = (
                intersection.find_colliding_cells(muscle_mesh, self.cutting_surface)
            )

            self.intersecting_cells = intersecting_cell_tags.indices[
                intersecting_cell_tags.values == 1
            ]
            self.colliding_muscle_cells = colliding_muscle_tags.indices[
                colliding_muscle_tags.values == 1
            ]

            self.integration_measure = ufl.Measure(
                "dx", domain=self.cutting_surface, subdomain_data=intersecting_cell_tags
            )

        elif method == "box":
            box_mesh, box_ft = intersection.create_midpoint_box(
                muscle_mesh,
                thickness=thickness,
                resolution_xyz=(2, resolution, resolution),
            )
            self.cutting_surface = box_mesh

            colliding_muscle_tags, intersecting_cell_tags = (
                intersection.find_colliding_cells(muscle_mesh, self.cutting_surface)
            )
            self.intersecting_cells = intersecting_cell_tags.indices[
                intersecting_cell_tags.values == 1
            ]
            self.colliding_muscle_cells = colliding_muscle_tags.indices[
                colliding_muscle_tags.values == 1
            ]

            # Create a new facet tag for the integration surface: facets that are BOTH
            # on the tagged face (marker 1) AND belong to an intersecting cell.
            tagged_facets = box_ft.indices[box_ft.values == 1]
            self.cutting_surface.topology.create_connectivity(3, 2)
            c_to_f = self.cutting_surface.topology.connectivity(3, 2)

            integration_facets = []
            for cell in self.intersecting_cells:
                cell_facets = c_to_f.links(cell)
                common_facets = np.intersect1d(
                    cell_facets, tagged_facets, assume_unique=True
                )
                integration_facets.extend(common_facets)

            integration_facets = np.unique(integration_facets).astype(np.int32)
            integration_facet_tags = dolfinx.mesh.meshtags(
                self.cutting_surface,
                2,
                integration_facets,
                np.full_like(integration_facets, 1),
            )

            self.integration_measure = ufl.Measure(
                "ds", domain=self.cutting_surface, subdomain_data=integration_facet_tags
            )
        else:
            raise ValueError(f"Unknown CSA method: {method}")

        print(f"CSA setup complete using '{method}' method.")

    def plot_csa_surface(self):
        """
        Visualizes the muscle geometry, the generated cutting surface, and highlights
        the cells on that surface that are actually intersecting the muscle.
        `setup_csa_surface()` must be called before this method.
        """
        if self.cutting_surface is None:
            raise RuntimeError(
                "`setup_csa_surface()` must be called before `plot_csa_surface()`."
            )

        import pyvista
        # Create a plotter
        plotter = pyvista.Plotter()
        plotter.add_text("Muscle with CSA Cutting Surface\n(Intersecting cells in yellow)", position='upper_left', font_size=10)

        # 1. Get the muscle mesh and convert to a PyVista grid
        muscle_mesh = self.domain
        topo_muscle, cell_types_muscle, geom_muscle = dolfinx.plot.vtk_mesh(muscle_mesh)
        grid_muscle = pyvista.UnstructuredGrid(topo_muscle, cell_types_muscle, geom_muscle)

        # Add the muscle mesh to the plotter with some transparency
        plotter.add_mesh(grid_muscle, style='surface', color='lightgray', opacity=0.5)

        # 2. Get the cutting surface mesh and convert to a PyVista grid
        topo_surface, cell_types_surface, geom_surface = dolfinx.plot.vtk_mesh(self.cutting_surface)
        grid_surface = pyvista.UnstructuredGrid(topo_surface, cell_types_surface, geom_surface)

        # Add the full cutting surface mesh to the plotter with a distinct color and some transparency
        plotter.add_mesh(grid_surface, color='red', show_edges=True, opacity=0.3)

        # 3. Extract and highlight the actual intersecting cells on the cutting surface
        if self.intersecting_cells is not None and len(self.intersecting_cells) > 0:
            intersecting_grid = grid_surface.extract_cells(self.intersecting_cells)
            plotter.add_mesh(intersecting_grid, color='yellow', show_edges=True)
        else:
            print("Warning: No intersecting cells found to highlight.")


        # 4. Show the plot
        plotter.view_isometric()
        plotter.show()

    def compute_csa(self, u=None):
        """
        Computes the cross-sectional area using the pre-computed cutting surface.
        `setup_csa_surface()` must be called before this method.

        Args:
            u (Function, optional): Displacement field (CG2). Defaults to None.

        Returns:
            float: Cross-sectional area.
        """
        if self.cutting_surface is None or self.integration_measure is None:
            raise RuntimeError(
                "`setup_csa_surface()` must be called before `compute_csa()`."
            )
      
        # --- Case 1: Undeformed CSA ---
        if u is None:
            # Integrate 1 over the pre-defined intersection measure (dx or ds)
            csa_local = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(1 * self.integration_measure(1))
            )
            return self.domain.comm.allreduce(csa_local, op=MPI.SUM)

        # --- Case 2: Deformed CSA ---

        # Method "plane": Interpolate F from muscle to plane
        if self.cutting_surface.topology.dim == 2:
            V_F_muscle = dolfinx.fem.functionspace(self.domain, ("DG", 1, (3, 3)))
            F_muscle = dolfinx.fem.Function(V_F_muscle)
            F_expr = dolfinx.fem.Expression(
                ufl.Identity(3) + ufl.grad(u), get_interpolation_points(V_F_muscle.element)
            )
            F_muscle.interpolate(F_expr)

            V_F_plane = dolfinx.fem.functionspace(
                self.cutting_surface, ("DG", 1, (3, 3))
            )
            F_plane = dolfinx.fem.Function(V_F_plane)

            all_cells =  np.arange(self.cutting_surface.topology.index_map(self.cutting_surface.topology.dim).size_local, dtype=np.int32)

            interp_data = dolfinx.fem.create_interpolation_data(
                V_F_plane, V_F_muscle, self.intersecting_cells
            )
            F_plane.interpolate_nonmatching(
                F_muscle, self.intersecting_cells, interp_data
                #F_muscle, all_cells, interp_data
            )

            J = ufl.det(F_plane)
            N = ufl.as_vector(self.plane_normal)
            f = ufl.inv(F_plane.T) * N
            norm = ufl.sqrt(ufl.dot(f, f))
            csa_local = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(J * norm * self.integration_measure(1))
            )

        # Method "box": Interpolate u from muscle to box, then compute F
        elif self.cutting_surface.topology.dim == 3:
            V_u_box = dolfinx.fem.functionspace(
                self.cutting_surface, ("Lagrange", 2, (3,))
            )
            u_box = dolfinx.fem.Function(V_u_box)
            interp_data = dolfinx.fem.create_interpolation_data(
                V_u_box, u.function_space, self.intersecting_cells
            )
            u_box.interpolate_nonmatching(u, self.intersecting_cells, interp_data)

            F_box = ufl.Identity(3) + ufl.grad(u_box)
            J = ufl.det(F_box)
            N = ufl.FacetNormal(self.cutting_surface)
            f = ufl.inv(F_box.T) * N
            norm = ufl.sqrt(ufl.dot(f, f))
            csa_local = dolfinx.fem.assemble_scalar(
                dolfinx.fem.form(J * norm * self.integration_measure(1))
            )

        else:
            raise ValueError("Cutting surface must be 2D or 3D")

        return self.domain.comm.allreduce(csa_local, op=MPI.SUM)


class CylinderGmsh(Geometry):
    def __init__(self):
        mesh_file = mesh_folder / "muscle-cylinder" / "muscle-cylinder.msh"
        mesh_data = dolfinx.io.gmsh.read_from_msh(
            mesh_file, MPI.COMM_WORLD, gdim=3
        )
        domain = mesh_data.mesh
        facet_markers = mesh_data.facet_tags

        self.name = "cylinder_gmsh"

        # Let fiber direction M be unit vector in z direction
        M = [0.0] * domain.geometry.dim
        M[2] = 1.0
        M = ufl.as_vector(M)

        # Call the super class constructor
        super().__init__(domain, M)

        # Facet tags
        self.ft = facet_markers


class Cylinder(Geometry):
    """Cylinder geometry with cylinder axis and fiber directions in z direction"""

    def __init__(
        self,
        radius: float = 1.0,
        length: float = 5.0,
        h: float = 0,
    ):
        """Cylinder geometry with given radius, length and mesh size h"""
        # Create cylinder mesh with Gmsh
        domain = self.construct_gmsh_cylinder(radius, length, h)

        self.name = "cylinder"

        # Let fiber direction M be unit vector in z direction
        M = [0.0] * domain.geometry.dim
        M[2] = 1.0
        M = ufl.as_vector(M)

        # Call the super class constructor
        super().__init__(domain, M)

        # Define locator functions for the ends of the cylinder
        def bottom(x):
            return np.isclose(x[2], 0.0)

        def top(x):
            return np.isclose(x[2], length)

        # Define markers for the boundaries
        self.boundaries = [
            Marker("bottom", 1, self.fdim, bottom),
            Marker("top", 2, self.fdim, top),
        ]
        # Create facet tags based on markers
        self.construct_facet_tags()

    def construct_gmsh_cylinder(self, radius, length, h):
        # TODO: embed the cross-section of the cylinder

        import gmsh

        if np.isclose(h, 0):
            h = 0.1 * radius

        # Create the cylinder mesh using gmsh
        gmsh.initialize()

        gmsh.model.add("cylinder")

        # add cylinder of given length and radius and center at (0,0,0)
        cylinder = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, length, radius)

        gmsh.model.occ.synchronize()
        gmsh.model.add_physical_group(3, [cylinder], tag=1)
        facets = gmsh.model.getBoundary([(3, cylinder)])
        gmsh.model.add_physical_group(2, [facets[0][1]], tag=2)

        f = gmsh.model.mesh.field.add("MathEval")
        gmsh.model.mesh.field.setString(f, "F", str(h))
        # Set the meshing field as the background field
        gmsh.model.mesh.field.setAsBackgroundMesh(f)
        # Synchronize to process the defined geometry
        gmsh.model.geo.synchronize()

        gmsh.model.mesh.generate(3)
        rank = MPI.COMM_WORLD.rank
        tmp_folder = Path(f"tmp_cylinder_{radius}_{rank}")
        tmp_folder.mkdir(exist_ok=True)
        gmsh_file = tmp_folder / "cylinder.msh"
        gmsh.write(str(gmsh_file))
        # gmsh.finalize()

        # return dolfin mesh of max dimension (parent mesh) and marker functions
        model_rank = 0
        mesh_data = dolfinx.io.gmsh.model_to_mesh(
            gmsh.model, MPI.COMM_WORLD, model_rank
        )
        domain = mesh_data.mesh

        # remove tmp mesh and tmp folder
        gmsh_file.unlink(missing_ok=False)
        tmp_folder.rmdir()

        return domain


class IdealizedFusiform(Geometry):

    def __init__(self, meshfile, fibersfile=None, comm=MPI.COMM_WORLD):

        # Read muscle mesh and facet tags from file
        with dolfinx.io.XDMFFile(comm, meshfile, "r") as f:
            domain = f.read_mesh(name="Grid")
            domain.topology.create_connectivity(
                domain.topology.dim, domain.topology.dim - 1
            )
            self.ft = f.read_meshtags(domain, name="ft")

        #tagfile = meshfile.with_name(meshfile.name.replace(".xdmf", "_ft.xdmf"))
        #with dolfinx.io.XDMFFile(comm, tagfile, "r") as f:
        #    self.ft = f.read_meshtags(domain, name="Grid")

        if fibersfile:
            V_fibers = dolfinx.fem.functionspace(
                domain,
                (
                    "Lagrange",
                    2,
                    (domain.geometry.dim,),
                ),
            )
            fibers = dolfinx.fem.Function(V_fibers, name="fiberdirection")
            adios4dolfinx.read_function(fibersfile, fibers)
        else:
            print(
                "No fiber orientation file provided. Calculating analytical fiber direction."
            )

            # Extract coordinates
            x = ufl.SpatialCoordinate(domain)
            x_coord = x[0]
            y_coord = x[1]
            z_coord = x[2]

            # Parameterize in terms of u = 5 * z_coord s.t. we work with u in [0, 1]
            u_param = 5 * z_coord

            # tolerances
            tol_small = 1e-8
            epsilon = 1e-12
            tol_axial_region = 1e-4

            # Define the piecewise derivatives y'(u)
            # Assuming a natural cubic spline for y.
            y_prime_u_seg1_expr = -6.0 * z_coord**2 + 0.06  # bottom segment
            y_prime_u_seg2_expr = 6.0 * z_coord**2 - 2.4 * z_coord + 0.18  # top segment

            # Conditional expression for y'(u)
            y_prime_u = ufl.conditional(
                ufl.le(u_param, 0.5 + tol_small),
                y_prime_u_seg1_expr,
                y_prime_u_seg2_expr,
            )

            # Define the piecewise derivative z'(u)
            z_prime_u_expr = 0.2

            # Radius in the xy-plane
            r_xy_squared = x_coord**2 + y_coord**2
            r_xy = ufl.sqrt(r_xy_squared)

            # Purely axial when at Z-axis
            is_at_z_axis = ufl.lt(r_xy, tol_axial_region)
            fiber_x_axial = 0.0
            fiber_y_axial = 0.0
            fiber_z_axial = 1.0

            # Avoid division by zero for the 'off-axis' calculation
            safe_r_xy_off_axis = ufl.conditional(ufl.lt(r_xy, epsilon), epsilon, r_xy)

            fiber_x_off_axis = (x_coord / safe_r_xy_off_axis) * y_prime_u
            fiber_y_off_axis = (y_coord / safe_r_xy_off_axis) * y_prime_u
            fiber_z_off_axis = z_prime_u_expr

            # Combine using ufl.conditional for the full unnormalized vector
            fiber_vector_unnormalized_x = ufl.conditional(
                is_at_z_axis, fiber_x_axial, fiber_x_off_axis
            )
            fiber_vector_unnormalized_y = ufl.conditional(
                is_at_z_axis, fiber_y_axial, fiber_y_off_axis
            )
            fiber_vector_unnormalized_z = ufl.conditional(
                is_at_z_axis, fiber_z_axial, fiber_z_off_axis
            )

            fiber_direction_ufl_expr = ufl.as_vector(
                [
                    fiber_vector_unnormalized_x,
                    fiber_vector_unnormalized_y,
                    fiber_vector_unnormalized_z,
                ]
            )

            # Normalization, avoiding division by zero
            norm_fiber_squared = ufl.dot(
                fiber_direction_ufl_expr, fiber_direction_ufl_expr
            )
            norm_fiber = ufl.sqrt(norm_fiber_squared)
            safe_norm_fiber = ufl.conditional(
                ufl.lt(norm_fiber, epsilon), epsilon, norm_fiber
            )
            fiber_direction_normalized = fiber_direction_ufl_expr / safe_norm_fiber

            # Create a vector function for the fiber directions and interpolate the expression
            V_fibers = dolfinx.fem.functionspace(
                domain,
                (
                    "Lagrange",
                    2,
                    (domain.geometry.dim,),
                ),
            )
            fibers = dolfinx.fem.Function(V_fibers, name="fiberdirection")
            fiber_expr = dolfinx.fem.Expression(
                fiber_direction_normalized, get_interpolation_points(V_fibers.element)
            )
            fibers.interpolate(fiber_expr)

        self.name = "idealized_fusiform"
        super().__init__(domain, fibers)
