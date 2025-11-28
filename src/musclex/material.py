import dolfinx
import basix
import ufl
import adios4dolfinx
import adios2
from mpi4py import MPI
import yaml
import musclex
import numpy as np
import time
from packaging.version import Version

is_dolfinx_10_or_newer = Version(dolfinx.__version__) >= Version("0.10.0")

from pathlib import Path
from musclex.utils import get_interpolation_points, mpiprint

tmpdir = Path(__file__).parents[1] / "results" / "tmp"

# Unique temporary directory for caching
# prevent race conditions when running multiple simulations in parallel
import tempfile
base_cache_dir = Path.home() / ".cache" / "fenics" 
base_cache_dir.mkdir(parents=True, exist_ok=True)
cache_dir = tempfile.mkdtemp(dir=base_cache_dir, prefix="fenics_job_")
jit_options = {"cache_dir": str(cache_dir)}


class MuscleRohrle:

    def __init__(
        self,
        domain: dolfinx.mesh.Mesh,
        ft: dolfinx.mesh.MeshTags,
        conf_file: str,
        fibers,
        output_dir: Path = tmpdir,
        clamp_type: str = "full",  # "full" or "z"
        pin_endpoint: bool = False, # whether to pin a point at one endpoint when using robin BCs
    ):
        self.domain = domain
        self.ft = ft
        self.a0 = fibers
        self.output_dir = output_dir
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.clamp_type = clamp_type
        self.pin_endpoint = pin_endpoint

        self.domain.topology.create_connectivity(
            self.domain.topology.dim - 1, self.domain.topology.dim
        )

        with open(conf_file) as config:
            conf = yaml.load(config, Loader=yaml.FullLoader)
            self.params = conf["material_parameters"]

        self.setup_fem()
        self.set_parameters()

    def setup_fem(self):
        """Set up finite element spaces, measures, and functions."""

        mpiprint("Setting up FEM...")
        tic = time.perf_counter()
        metadata = {"quadrature_degree": 4}
        self.dx = ufl.Measure("dx", domain=self.domain, metadata=metadata)
        self.ds = ufl.Measure("ds", self.domain, subdomain_data=self.ft)
        self.dS = ufl.Measure("dS", self.domain, subdomain_data=self.ft)

        # Construct mixed element
        gdim = self.domain.geometry.dim
        P2 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 2, shape=(gdim,))
        P1 = basix.ufl.element("Lagrange", self.domain.basix_cell(), 1)
        mixed_element = basix.ufl.mixed_element([P2, P1])

        # Function space
        self.state_space = dolfinx.fem.functionspace(self.domain, mixed_element)
        mpiprint(
            f"dofs: { self.state_space.dofmap.index_map.size_global * self.state_space.dofmap.index_map_bs}"
        )

        # Collapsed subspaces and dofmaps
        self.V, self.V_to_W = self.state_space.sub(0).collapse()
        self.Q, self.Q_to_W = self.state_space.sub(1).collapse()

        # DG0 function space for fiber stretch
        self.W = dolfinx.fem.functionspace(self.domain, ("DG", 0))

        # self.trial_state = ufl.TrialFunctions(self.state_space)
        self.state = dolfinx.fem.Function(self.state_space, name="state")
        self.test_state = ufl.TestFunctions(self.state_space)

        # Trial and test functions for displacement u and pressure p
        self.u, self.p = ufl.split(self.state)
        self.u_sol, self.p_sol = self.state.split()
        self.v, self.q = self.test_state

        # Initialize activation level to 0
        self.alpha = dolfinx.fem.Constant(self.domain, 0.0)

        mpiprint(f"Setup FEM finished in {time.perf_counter() - tic:.2f} seconds.\n")

    def prepare_output(self):
        """Prepare output functions and files"""
        self.u_out = dolfinx.fem.Function(self.V, name="u")
        self.p_out = dolfinx.fem.Function(self.Q, name="p")

        # Define DG0 function for fiber stretch
        self.lmbda_out = dolfinx.fem.Function(self.W, name="Fiber stretch")

        self.u_outfile = dolfinx.io.VTXWriter(
            self.domain.comm, self.output_dir / "rohrle_u.bp", self.u_out, engine="BP4"
        )
        self.p_outfile = dolfinx.io.VTXWriter(
            self.domain.comm, self.output_dir / "rohrle_p.bp", self.p_out, engine="BP4"
        )
        self.lmbda_outfile = dolfinx.io.VTXWriter(
            self.domain.comm,
            self.output_dir / "rohrle_fiberstretch.bp",
            self.lmbda_out,
            engine="BP4",
        )

        self.outfile_pp = self.output_dir / "output_pp.bp"
        adios4dolfinx.write_mesh(self.outfile_pp, self.domain)
        adios4dolfinx.write_meshtags(
            self.outfile_pp, self.domain, self.ft, meshtag_name="facet_tags"
        )
        V = dolfinx.fem.functionspace(
            self.domain, ("Lagrange", 2, (self.domain.geometry.dim,))
        )
        if not isinstance(self.a0, dolfinx.fem.Function):
            fibers = dolfinx.fem.Function(V, name="fiberdirection")
            # Interpolate the fiber expression to the function space
            fibers_expr = dolfinx.fem.Expression(
                self.a0, get_interpolation_points(V.element), self.domain.comm
            )
            fibers.interpolate(fibers_expr)
        else:
            fibers = self.a0

        adios4dolfinx.write_function(self.outfile_pp, fibers, time=0.0)

    def write_output(self, a):
        """Write output functions to file"""
        self.u_outfile.write(a)
        self.p_outfile.write(a)
        self.lmbda_outfile.write(a)

        adios4dolfinx.write_function(self.outfile_pp, self.u_out, time=a)
        adios4dolfinx.write_function(self.outfile_pp, self.p_out, time=a)
        mpiprint(f"Output written for activation level {a:.3f} to {self.outfile_pp}")

    def set_parameters(self):
        """Set material parameters"""
        self.c1 = dolfinx.fem.Constant(self.domain, self.params["c1"])
        self.c2 = dolfinx.fem.Constant(self.domain, self.params["c2"])
        self.sigma_pass_ff = dolfinx.fem.Constant(
            self.domain, self.params["sigma_pass_ff"]
        )
        self.sigma_act_ff = dolfinx.fem.Constant(
            self.domain, self.params["sigma_act_ff"]
        )
        self.lmbda_0 = dolfinx.fem.Constant(self.domain, self.params["lmbda_0"])
        self.g1 = dolfinx.fem.Constant(self.domain, self.params["g1"])
        self.g2 = dolfinx.fem.Constant(self.domain, self.params["g2"])
        self.k_tendon = dolfinx.fem.Constant(self.domain, self.params["k_tendon"])

    def clamp_ends_mixed(self):
        """Clamp all components of the displacement at the two ends of the muscle."""
        # Full clamp (zero displacement, all components) at both ends
        # assuming the two muscle ends are already tagged with 1 and 2
        u_zero = dolfinx.fem.Function(self.V)

        self.domain.topology.create_connectivity(self.ft.dim, self.domain.topology.dim)
        # Find dofs on the two ends
        clamped_dofs_1 = dolfinx.fem.locate_dofs_topological(
            (self.state_space.sub(0), self.V),
            self.ft.dim,
            self.ft.find(1),
        )
        clamped_dofs_2 = dolfinx.fem.locate_dofs_topological(
            (self.state_space.sub(0), self.V),
            self.ft.dim,
            self.ft.find(2),
        )
        print(f"Number of DOFs found for tag 2: {len(clamped_dofs_2)}")
        print(f"Number of DOFs found for tag 1: {len(clamped_dofs_1)}")

        # Define BCs
        bcs = [
            dolfinx.fem.dirichletbc(u_zero, clamped_dofs_1, self.state_space.sub(0)),
            dolfinx.fem.dirichletbc(u_zero, clamped_dofs_2, self.state_space.sub(0)),
        ]
        return bcs

    def clamp_axial_and_pin_point(self):
        """
        Applies boundary conditions to simulate an isometric contraction.
        1. Clamps the axial (Z) displacement on both end faces.
        2. Clamps the transverse (X, Y) displacement on a single point
           to prevent rigid body motion.
        """
        bcs = []

        # --- Part 1: Constrain axial (z) displacement on ends ---

        # Isolate function space for z-component of displacement
        z_comp_space, _ = self.state_space.sub(0).sub(2).collapse()
        u_zero_scalar = dolfinx.fem.Function(z_comp_space)
        u_zero_scalar.x.array[:] = 0.0

        # Find DOFs for z-component on bottom end (tag 1)
        clamped_dofs_z_1 = dolfinx.fem.locate_dofs_topological(
            (self.state_space.sub(0).sub(2), z_comp_space), self.ft.dim, self.ft.find(1)
        )
        bc_z1 = dolfinx.fem.dirichletbc(
            u_zero_scalar, clamped_dofs_z_1, self.state_space.sub(0).sub(2)
        )
        bcs.append(bc_z1)

        # Find DOFs for z-component on top end (tag 2)
        clamped_dofs_z_2 = dolfinx.fem.locate_dofs_topological(
            (self.state_space.sub(0).sub(2), z_comp_space), self.ft.dim, self.ft.find(2)
        )
        bc_z2 = dolfinx.fem.dirichletbc(
            u_zero_scalar, clamped_dofs_z_2, self.state_space.sub(0).sub(2)
        )
        bcs.append(bc_z2)

        # --- Part 2: Constrain Points ---
        # Find the N vertices on each end face (tags 1 and 2) closest to the
        # centroid of that face.
        N = 10
        for tag in [1, 2]:
            end_face_facets = self.ft.find(tag)
            self.domain.topology.create_connectivity(self.domain.topology.dim - 1, 0)
            end_face_vertices = dolfinx.mesh.compute_incident_entities(
                self.domain.topology, end_face_facets, self.domain.topology.dim - 1, 0
            )
            end_face_vertices = np.unique(end_face_vertices)

            # Get coordinates of all vertices on the end face
            end_face_coords = self.domain.geometry.x[end_face_vertices]

            # Calculate the centroid of the end face
            centroid = np.mean(end_face_coords, axis=0)
            # Find the vertices on the end face closest to the centroid
            distances = np.linalg.norm(end_face_coords - centroid, axis=1)
            closest_vertex_local_idxs = np.argsort(distances)[:N]
            closest_vertex_global_idxs = end_face_vertices[closest_vertex_local_idxs]

            pin_vertex = np.array(closest_vertex_global_idxs, dtype=np.int32)

            point_coords = self.domain.geometry.x[pin_vertex]
            mpiprint(
                f"Pinning point at vertex index {pin_vertex} with coordinates {point_coords}."
            )
            # Create 0-3 connectivity
            self.domain.topology.create_connectivity(0, self.domain.topology.dim)

            # Find displacement DOF of chosen vertices
            point_dof = dolfinx.fem.locate_dofs_topological(
                (self.state_space.sub(0), self.V), 0, pin_vertex
            )
            # Set dofs to zero displacement
            u_zero = dolfinx.fem.Function(self.V)
            bc_pin = dolfinx.fem.dirichletbc(u_zero, point_dof, self.state_space.sub(0))

            bcs.append(bc_pin)

        return bcs

    def clamp_robin(self, R):
        """
        Applies Robin boundary conditions to represent elastic tendons at the muscle
        ends (surfaces tagged 1 and 2). Also pins one point to prevent rigid 
        body motion.

        Args:
            R: The UFL form for the residual of the weak form.

        Returns:
            tuple: A tuple containing:
                - R (ufl.Form): The modified residual with the Robin term added.
                - bcs (list): A list containing the Dirichlet BC for pinning.
        """
        N = ufl.FacetNormal(self.domain)
        F = ufl.grad(self.u) + ufl.Identity(3)
        J = ufl.det(F)
        cof = J * ufl.inv(F).T
        cofnorm = ufl.sqrt(ufl.dot(cof * N, cof * N))
        NN = 1 / cofnorm * cof * N # unit normal in current configuration
        nn = ufl.outer(NN, NN)
        value = -nn * self.k_tendon * self.u
        # Add the Robin boundary term to the weak form (residual).
        R += - ufl.dot(value, self.v) * cofnorm * self.ds(1) \
             - ufl.dot(value, self.v) * cofnorm * self.ds(2)
        mpiprint(f"Applying Robin BC with k={self.k_tendon.value}.")

        # In addition, pin a point at one end to prevent rigid body motion
        # This is needed for the idealized geometry where Robin BCs on the flat
        # end surfaces do not fully constrain the muscle.
        if self.pin_endpoint:
            N = 1 # number of points to pin
            bcs = []
            for tag in [1]: # pin only at one end
                end_face_facets = self.ft.find(tag)
                self.domain.topology.create_connectivity(self.domain.topology.dim - 1, 0)
                end_face_vertices = dolfinx.mesh.compute_incident_entities(
                    self.domain.topology, end_face_facets, self.domain.topology.dim - 1, 0
                )
                end_face_vertices = np.unique(end_face_vertices)

                # Get coordinates of all vertices on the end face
                end_face_coords = self.domain.geometry.x[end_face_vertices]

                # Calculate the centroid of the end face
                centroid = np.mean(end_face_coords, axis=0)
                # Find the vertices on the end face closest to the centroid
                distances = np.linalg.norm(end_face_coords - centroid, axis=1)
                closest_vertex_local_idxs = np.argsort(distances)[:N]
                closest_vertex_global_idxs = end_face_vertices[closest_vertex_local_idxs]

                pin_vertex = np.array(closest_vertex_global_idxs, dtype=np.int32)

                point_coords = self.domain.geometry.x[pin_vertex]
                mpiprint(
                    f"Pinning point at vertex index {pin_vertex} with coordinates {point_coords}."
                )
                # Create 0-3 connectivity
                self.domain.topology.create_connectivity(0, self.domain.topology.dim)

                # Find displacement DOF of chosen vertices
                point_dof = dolfinx.fem.locate_dofs_topological(
                    (self.state_space.sub(0), self.V), 0, pin_vertex
                )
                # Set dofs to zero displacement
                u_zero = dolfinx.fem.Function(self.V)
                bc_pin = dolfinx.fem.dirichletbc(u_zero, point_dof, self.state_space.sub(0))

                bcs.append(bc_pin)
            return R, bcs
        else:
            return R, []

        return R

    def force_active(self, I4):
        """Normalized active fiber force. Given as a piecewise polynomial"""
        l = ufl.sqrt(I4) / self.lmbda_0
        return ufl.conditional(
            ufl.le(l, 0.4),
            0,
            ufl.conditional(
                ufl.le(l, 0.6),
                9 * (l - 0.4) ** 2,
                ufl.conditional(
                    ufl.le(l, 1.4),
                    1 - 4 * (1 - l) ** 2,
                    ufl.conditional(ufl.le(l, 1.6), 9 * (l - 1.6) ** 2, 0),
                ),
            ),
        )

    def force_passive(self, I4):
        """Normalized passive fiber force. Given as a piecewise exponential function."""
        l = ufl.sqrt(I4) / self.lmbda_0
        f_p1 = self.g1 * (ufl.exp(self.g2 * (l - 1)) - 1)
        f_p2 = (self.g1 * self.g2 * ufl.exp(0.4 * self.g2)) * l + self.g1 * (
            (1 - 1.4 * self.g2) * ufl.exp(0.4 * self.g2) - 1
        )
        return ufl.conditional(
            ufl.le(l, 1),
            0.0,
            ufl.conditional(
                ufl.le(l, 1.4),
                f_p1,
                f_p2,
            ),
        )

    def stress_PK2(self, alpha, F, p):
        """Second Piola-Kirchhoff stress tensor.

        (See Rohrle et al 2012, eq 11)"""

        # Kinematics
        J = ufl.det(F)  # Jacobian
        C = F.T * F  # right Cauchy-Green deformation tensor
        I1 = ufl.tr(C)  # first invariant
        I4 = ufl.inner(C * self.a0, self.a0)  # fourth invariant (squared fiber stretch)
        lmbda = ufl.sqrt(I4)  # fiber stretch
        self.lmbda_expr = dolfinx.fem.Expression(
            lmbda, get_interpolation_points(self.W.element), comm=MPI.COMM_WORLD
        )

        #  1. Mooney-Rivlin model
        S_mooneyrivlin = self.c1 * ufl.Identity(self.domain.geometry.dim) + self.c2 * (
            I1 * ufl.Identity(self.domain.geometry.dim) - C
        )

        #  2. Near incompressibility term
        S_vol = -p * J * ufl.inv(C)

        #  3. Anisotropic passive stress (along-fiber)
        S_anisotropic = (
            self.sigma_pass_ff
            / I4
            * self.force_passive(I4)
            * ufl.outer(self.a0, self.a0)
        )

        #  4. Anisotropic active stress (along-fiber)
        S_active = (
            alpha
            * self.sigma_act_ff
            / I4
            * self.force_active(I4)
            * ufl.outer(self.a0, self.a0)
        )
        return S_mooneyrivlin + S_vol + S_anisotropic + S_active

    def stress_PK1(self, S, F):
        """Convert second Piola-Kirchhoff stress tensor to first Piola-Kirchhoff stress tensor"""
        return F * S

    def stress_Cauchy(self, P, F):
        """Convert first Piola-Kirchhoff stress tensor to Cauchy stress tensor"""
        return 1 / ufl.det(F) * P * F.T
        # return 1 / ufl.det(self.Fe) * self.stress_PK1 * self.Fe.T

    def stress_VM(self, P, F):
        """Compute Von Mises stress, a scalar measure of stress intensity."""
        sigma = self.stress_Cauchy(P, F)
        sigma_dev = sigma - (1 / 3) * ufl.tr(sigma) * ufl.Identity(len(self.u))
        sigma_vm = ufl.sqrt((3 / 2) * ufl.inner(sigma_dev, sigma_dev))
        return sigma_vm

    def reaction_force(self, sigma):
        """Compute reaction force in normal direction at muscle end."""
        n = ufl.FacetNormal(self.domain)
        Fr_local = dolfinx.fem.assemble_scalar(
            dolfinx.fem.form(ufl.dot(ufl.dot(sigma, n), n) * self.ds(1))
        )
        return self.domain.comm.allreduce(Fr_local, op=MPI.SUM)

    def solve(
        self,
        activation_levels=np.array([0.0]),
        Fg=None,
    ):
        """Solve the mechanical model for a given set of activation levels

        Args:
            activation_levels (np.array): Activation levels. Defaults to [0.0].
            Fg (ufl tensor, optional): Growth tensor. Defaults to None.

        Returns:
            n (int): Number of nonlinear solver iterations for final activation level.
            converged (bool): Convergence status for final activation level.
            reaction_forces (list): Reaction forces for each activation level.
        """
        self.prepare_output()

        ## --------- KINEMATICS --------- ##
        I = ufl.Identity(self.domain.geometry.dim)
        F = I + ufl.grad(self.u)

        # If a growth tensor is given, we solve for elastic deformation only
        if Fg:
            Fe = F * ufl.inv(Fg)
        else:
            Fe = F

        Je = ufl.det(Fe)

        ## --------- MATERIAL MODEL --------- ##

        S = self.stress_PK2(self.alpha, Fe, self.p)  # Second Piola-Kirchhoff stress
        P = self.stress_PK1(S, Fe)  # First Piola-Kirchhoff stress
        sigma = self.stress_Cauchy(P, Fe)  # Cauchy stress


        ## --------- DEFINE RESIDUAL --------- ##
        R = ufl.inner(ufl.grad(self.v), P) * self.dx + self.q * (Je - 1) * self.dx

        ## --------- BOUNDARY CONDITIONS --------- ##
        if self.clamp_type == "full":
            bcs = self.clamp_ends_mixed()
        elif self.clamp_type == "z":
            bcs = self.clamp_axial_and_pin_point()
        elif self.clamp_type == "robin":
            R, bcs = self.clamp_robin(R)
        elif self.clamp_type == "none":
            bcs = []
        else:
            raise ValueError(
                f"Unknown clamp_type '{self.clamp_type}'. Choose from 'full', 'z', 'robin', or 'none'."
            )
        mpiprint(f"Using '{self.clamp_type}' clamp type.")

        ## --------- SOLVE --------- ##


        # Set up non-linear solver
        
        if is_dolfinx_10_or_newer:
            petsc_options_nonlinear = {
                "ksp_type": "preonly",
                "pc_type": "lu",
                "pc_factor_mat_solver_type": "mumps",
                "snes_atol": 1e-8,
                "snes_rtol": 1e-8,
                "snes_stol": 1e-10,
                "snes_max_it": 20,
                "snes_type": "newtonls",
                "snes_linesearch_type": "none",
                "snes_monitor": None, # monitor for nonlinear solver
                "ksp_monitor": None, # monitor for linear solver
            }
            solver = dolfinx.fem.petsc.NonlinearProblem(
                R,
                self.state,
                petsc_options_prefix="tmp_",
                petsc_options=petsc_options_nonlinear,
                bcs=bcs,
                jit_options=jit_options,
            )

            def monitor(ksp, its, rnorm):
                mpiprint(f"Iteration {its} residual: {rnorm}")

            solver.solver.setMonitor(monitor)
            
            if self.clamp_type == "none":
                from musclex.solvers import build_nullspace
                nullspace = build_nullspace(self.state_space)
                solver.A.setNullSpace(nullspace)
        else:
            solver = musclex.solvers.NewtonSolver(
                R, self.state, bcs, set_nullspace=False
            )

        # Initialize reaction force array
        reaction_forces = []

        # Solve for each activation level
        mpiprint("Starting simulation...")
        tic = time.perf_counter()
        for i, a in enumerate(activation_levels):
            mpiprint(f"{"="*50}\nSolving for activation level {a}")
            self.alpha.value = a  # set activation level

            # Solve the nonlinear problem
            # n, converged = solver.solve(self.state)

            dolfinx.log.set_log_level(dolfinx.log.LogLevel.INFO)
            if is_dolfinx_10_or_newer:
                solver.solve()
                converged = solver.solver.getConvergedReason() > 0
            else:
                n, converged = solver.solve(self.state)
            dolfinx.log.set_log_level(dolfinx.log.LogLevel.WARNING)

            if not converged:
                mpiprint(f"Warning: Solver did not converge for activation level {a}.")
                mpiprint("Stopping simulation.")
                break

            # Compute reaction force in normal direction at muscle end
            F_reac = self.reaction_force(sigma)
            mpiprint(f"F_reac = {F_reac}")
            reaction_forces.append(F_reac)

            # Update output functions
            self.u_out.x.array[:] = self.state.x.array[self.V_to_W]
            self.p_out.x.array[:] = self.state.x.array[self.Q_to_W]
            self.lmbda_out.interpolate(self.lmbda_expr)

            # Write output
            self.write_output(a)

        mpiprint(f"Simulation finished in {time.perf_counter() - tic:.2f} seconds.\n")

        return converged, reaction_forces

    @property
    def ndofs(self):
        """Helper to get total number of dofs in the mixed function space."""
        return (
            self.state_space.dofmap.index_map.size_global
            * self.state_space.dofmap.index_map_bs
        )

    @property
    def ncells(self):
        """Helper to get total number of cells in the mesh."""
        return self.domain.topology.index_map(self.domain.topology.dim).size_global
