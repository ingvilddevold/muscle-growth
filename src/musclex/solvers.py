import dolfinx
import dolfinx.fem.petsc
import dolfinx.nls.petsc
import ufl

import numpy as np

from mpi4py import MPI
from petsc4py import PETSc
from abc import ABC


class MechanicsSolver(ABC):
    def __init__(self, L, u, bcs):
        self.L = L
        self.bcs = bcs
        self.V = u.function_space
        self.domain = self.V.mesh
        self.build_nullspace()
        self.solver = None
        self.activations = None

    def solve(self, u):
        n, converged = self.solver.solve(u)
        u.x.scatter_forward()
        return n, converged

    def build_nullspace(self):
        # Build PETSc nullspace for 3D elasticity
        V = self.V
        # if V is mixed space, extract the first component
        if V.num_sub_spaces == 2:  # FIXME: will break in 2D
            V, _ = V.sub(0).collapse()

        # Create vectors that will span the nullspace
        bs = V.dofmap.index_map_bs
        length0 = V.dofmap.index_map.size_local
        basis = [
            dolfinx.la.vector(V.dofmap.index_map, bs=bs, dtype=PETSc.ScalarType)
            for i in range(6)
        ]
        b = [b.array for b in basis]

        # Get dof indices for each subspace (x, y and z dofs)
        dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

        # Set the three translational rigid body modes
        for i in range(3):
            b[i][dofs[i]] = 1.0

        # Set the three rotational rigid body modes
        x = V.tabulate_dof_coordinates()
        dofs_block = V.dofmap.list.flatten()
        x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
        b[3][dofs[0]] = -x1
        b[3][dofs[1]] = x0
        b[4][dofs[0]] = x2
        b[4][dofs[2]] = -x0
        b[5][dofs[2]] = x1
        b[5][dofs[1]] = -x2

        # Orthonormalize the six basis vectors
        _basis = [x._cpp_object for x in basis]
        dolfinx.cpp.la.orthonormalize(_basis)
        # assert dolfinx.cpp.la.is_orthonormal(_basis)

        basis_petsc = [
            PETSc.Vec().createWithArray(x[: bs * length0], bsize=3, comm=V.mesh.comm)
            for x in b
        ]
        self.nullspace = PETSc.NullSpace().create(vectors=basis_petsc)


class NonlinearPDE_SNESProblem:
    def __init__(
        self,
        F: ufl.form.Form,
        u: dolfinx.fem.function,
        bcs: list[dolfinx.fem.DirichletBC] = [],
    ):
        """Initialize SNES problem for nonlinear PDE.

        Args:
            F: The PDE residual F(u,v)
            u: The unknown
            bcs: List of Dirichlet boundary conditions
            J: UFL representation of the Jacobian (optional)
        """
        self.V = u.function_space
        du = ufl.TrialFunction(self.V)
        self.L = dolfinx.fem.form(F)
        self.a = dolfinx.fem.form(ufl.derivative(F, u, du))
        self.bc = bcs
        self._F, self._J = None, None
        self.u = u

    def F(self, snes, x, F):
        """Assemble residual vector."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import apply_lifting, assemble_vector, set_bc

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(
            addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD
        )

        with F.localForm() as f_local:
            f_local.set(0.0)
        assemble_vector(F, self.L)
        apply_lifting(F, [self.a], bcs=[self.bc], x0=[x], alpha=-1.0)
        F.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
        set_bc(F, self.bc, x, -1.0)

    def J(self, snes, x, J, P):
        """Assemble Jacobian matrix."""
        from petsc4py import PETSc

        from dolfinx.fem.petsc import assemble_matrix

        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)
        x.copy(self.u.x.petsc_vec)
        self.u.x.petsc_vec.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        J.zeroEntries()
        assemble_matrix(J, self.a, bcs=self.bc)
        J.assemble()

class SNESSolver(MechanicsSolver):

    def __init__(self, L, u, bcs, set_nullspace=True):
        super().__init__(L, u, bcs)

        self.problem = NonlinearPDE_SNESProblem(L, u, bcs)

        b = dolfinx.la.create_petsc_vector(
            self.problem.V.dofmap.index_map, self.problem.V.dofmap.index_map_bs
        )
        J = dolfinx.fem.petsc.create_matrix(self.problem.a)
        if set_nullspace:
            J.setNullSpace(self.nullspace)  # set rigid body motion null space

        snes = PETSc.SNES().create()
        snes.setFunction(self.problem.F, b)
        snes.setJacobian(self.problem.J, J)
        snes.setTolerances(atol=1.0e-10, rtol=1.0e-10, max_it=20)

        snes.getKSP().setType("preonly")
        snes.getKSP().setTolerances(rtol=1.0e-9)
        snes.getKSP().getPC().setType("lu")
        snes.getKSP().getPC().setFactorSolverType("mumps")

        snes.setMonitor(lambda _, it, residual: print("It:", it, "Residual:", residual))
        
        opts = PETSc.Options()
        opts["snes_linesearch_type"] = "basic"  # no line search
        # opts["snes_linesearch_monitor"] = None
        snes.setFromOptions()

        self.solver = snes

    def solve(self, u):
        # For SNES line search to function correctly it is necessary that the
        # u.x.petsc_vec in the Jacobian and residual is *not* passed to
        # snes.solve.
        x = u.x.petsc_vec.copy()
        x.ghostUpdate(addv=PETSc.InsertMode.INSERT, mode=PETSc.ScatterMode.FORWARD)

        self.solver.solve(None, x)
        n = self.solver.its
        converged = self.solver.is_converged
        u.x.scatter_forward()
        return n, converged


class NewtonSolver(MechanicsSolver):
    def __init__(self, L, u, bcs, set_nullspace=True):
        super().__init__(L, u, bcs)
        self.problem = dolfinx.fem.petsc.NonlinearProblem(L, u, bcs)
        solver = dolfinx.nls.petsc.NewtonSolver(
            u.function_space.mesh.comm, self.problem
        )
        solver.atol = 5e-5
        solver.rtol = 5e-8
        solver.convergence_criterion = "incremental"
        solver.max_it = 20
        solver.error_on_nonconvergence = False
        ksp = solver.krylov_solver
        ksp.setType("preonly")
        ksp.setTolerances(rtol=1.0e-13, atol=1.0e-13)
        ksp.getPC().setType("lu")
        ksp.getPC().setFactorSolverType("mumps")

        if set_nullspace:
            # Set rigid body motion null space
            solver.A.setNullSpace(self.nullspace)

        self.solver = solver


class SciFemSolver(MechanicsSolver):
    def __init__(self, R, K, states, bcs):
        import scifem

        petsc_options = {
            "ksp_type": "preonly",
            "pc_type": "lu",
            "pc_factor_mat_solver_type": "mumps",
        }
        solver = scifem.NewtonSolver(
            R,
            K,
            states,
            max_iterations=25,
            bcs=bcs,
            petsc_options=petsc_options,
        )

        def pre_solve(solver: scifem.NewtonSolver):
            print(f"Starting solve with {solver.max_iterations} iterations")

        def post_solve(solver: scifem.NewtonSolver):
            print(f"Solve completed with correction norm {solver.dx.norm(0)}")

        solver.set_pre_solve_callback(pre_solve)
        solver.set_post_solve_callback(post_solve)

        # FIXME: How to set nullspace for u only?
        # self.V = states[0].function_space  # assuming u is the first state
        # self.build_nullspace()
        # solver.A.setNullSpace(self.nullspace)

        self.solver = solver

    def solve(self):
        self.solver.solve()
        n = self.solver._solver.its
        converged = self.solver._solver.is_converged
        return n, converged


def build_nullspace_basis(
    V: dolfinx.fem.FunctionSpace, dtype=np.float64
) -> list[dolfinx.la.Vector]:
    """Build nullspace for 3D elasticity problems (rigid body modes)"""

    # Create vectors that will span the nullspace
    bs = V.dofmap.index_map_bs
    basis = [
        dolfinx.la.vector(V.dofmap.index_map, bs=bs, dtype=dtype) for i in range(6)
    ]
    b = [b.array for b in basis]

    # Get dof indices for each subspace (x, y and z dofs)
    dofs = [V.sub(i).dofmap.list.flatten() for i in range(3)]

    # Set the three translational rigid body modes
    for i in range(3):
        b[i][dofs[i]] = 1.0

    # Set the three rotational rigid body modes
    x = V.tabulate_dof_coordinates()
    dofs_block = V.dofmap.list.flatten()
    x0, x1, x2 = x[dofs_block, 0], x[dofs_block, 1], x[dofs_block, 2]
    b[3][dofs[0]] = -x1
    b[3][dofs[1]] = x0
    b[4][dofs[0]] = x2
    b[4][dofs[2]] = -x0
    b[5][dofs[2]] = x1
    b[5][dofs[1]] = -x2

    dolfinx.la.orthonormalize(basis)
    return basis


def build_nullspace(W):
    # Assuming W is a mixed space with displacement as first subspace
    V, V_to_W = W.sub(0).collapse()
    num_dofs_local = V.dofmap.index_map.size_local * V.dofmap.index_map_bs
    V_to_W = V_to_W[:num_dofs_local]
    basis = build_nullspace_basis(V)
    petsc_basis = []
    for b in basis:
        wh = dolfinx.fem.petsc.create_vector(W)
        wh.array_w[V_to_W] = b.array[:num_dofs_local]
        petsc_basis.append(wh)

    ns = PETSc.NullSpace().create(vectors=petsc_basis)
    return ns

