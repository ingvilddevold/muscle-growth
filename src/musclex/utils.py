from dolfinx import fem
from packaging.version import Version
import numpy as np
from mpi4py import MPI
import ufl
import dolfinx
from petsc4py import PETSc


def mpiprint(*arg):
    """Print only from rank 0."""
    if MPI.COMM_WORLD.rank == 0:
        print(*arg)


def get_interpolation_points(element):
    """
    Get interpolation points from a FEniCSx element,
    handling API changes before and after Dolfinx v0.10.
    """
    if Version(dolfinx.__version__) < Version("0.10.0"):
        return element.interpolation_points()
    else:
        return element.interpolation_points


def eval_expression(domain, expr, point):
    """Evaluate a UFL expression at a point."""
    # Determine what process owns a point and what cells it lies within

    tol = 1e-6
    if dolfinx.__version__ == "0.8.0":
        src_ranks, _, points, cells = dolfinx.cpp.geometry.determine_point_ownership(
            domain._cpp_object, point, tol
        )
        owning_points = np.array(points).reshape(-1, 3)
    elif Version(dolfinx.__version__) >= Version("0.9.0.0"):
        collision_data = dolfinx.cpp.geometry.determine_point_ownership(
            domain._cpp_object, point, tol
        )
        owning_points = collision_data.dest_points
        cells = collision_data.dest_cells
    else:
        raise NotImplementedError(f"Unsupported dolfinx version: {dolfinx.__version__}")

    # Pull owning points back to reference cell
    mesh_nodes = domain.geometry.x
    cmap = domain.geometry.cmap  # s[0]
    ref_x = np.zeros((len(cells), domain.geometry.dim), dtype=domain.geometry.x.dtype)
    for i, (point, cell) in enumerate(zip(owning_points, cells)):
        geom_dofs = domain.geometry.dofmap[cell]
        ref_x[i] = cmap.pull_back(point.reshape(-1, 3), mesh_nodes[geom_dofs])
    if len(cells) > 0:
        # NOTE: Expression lives on only this communicator rank
        d_expr = fem.Expression(expr, ref_x, comm=MPI.COMM_SELF)
        values = d_expr.eval(domain, np.asarray(cells).astype(np.int32))
        return values


def check_fibers(fibers):
    # Calculate maximum and minimum value of the fiber magnitude
    u1 = fibers.sub(0).collapse().x.array
    u2 = fibers.sub(1).collapse().x.array
    u3 = fibers.sub(2).collapse().x.array
    fibers_mag = np.sqrt(u1**2 + u2**2 + u3**2)

    print("Max. fiber magnitude: ", fibers_mag.max())
    print("Min. fiber magnitude: ", fibers_mag.min())


def check_expression_for_nans(domain, expr):
    for i in range(1, domain.geometry.x.shape[0]):
        point = domain.geometry.x[i, :]
        value = eval_expression(domain, expr, point)
        assert not np.isnan(value)


def kinematics_sanity_check(domain, u, F, dx, I1, I4, lmbda, J):
    # Quick sanity check for current displacement guess
    vol = fem.assemble_scalar(fem.form(fem.Constant(domain, 1.0) * dx))
    # print("norm u", u.x.norm())
    print("F norm squared", fem.assemble_scalar(fem.form(ufl.inner(F, F) * dx)) / vol)
    print("I1", fem.assemble_scalar(fem.form(I1 * dx)) / vol)  # expect d
    print("I4", fem.assemble_scalar(fem.form(I4 * dx)) / vol)  # expect 1
    print("lmbda", fem.assemble_scalar(fem.form(lmbda * dx)) / vol)  # expect 1
    print("J", fem.assemble_scalar(fem.form(J * dx)) / vol)  # expect 1


def interpolate_scalar_function(f, domain, name=None):
    """Interpolate a scalar UFL expression into a P2 function."""
    Q = fem.functionspace(domain, ("Lagrange", 2))
    expr = fem.Expression(f, get_interpolation_points(Q.element))
    func = fem.Function(Q, name=name)
    func.interpolate(expr)
    return func


def domain_volume(domain, dx):
    """Volume of the domain."""
    return fem.assemble_scalar(fem.form(fem.Constant(domain, 1.0) * dx))


def avg_scalar_expr(expr, domain, dx):
    "Average of a scalar expression over the domain."
    return fem.assemble_scalar(fem.form(expr * dx)) / domain_volume(domain, dx)


def project(expr, V_target, bcs=[]):
    """
    Project a ufl expression expr into a target function space V_target.
    """
    # Define target function
    target_func = fem.Function(V_target)

    # Define variational problem for projection
    dx = ufl.dx(V_target.mesh)
    w = ufl.TestFunction(V_target)
    v = ufl.TrialFunction(V_target)
    a = fem.form(ufl.inner(v, w) * dx)
    L = fem.form(ufl.inner(expr, w) * dx)

    # Assemble linear system
    A = fem.petsc.assemble_matrix(a, bcs)
    A.assemble()
    b = fem.petsc.assemble_vector(L)
    fem.petsc.apply_lifting(b, [a], [bcs])
    b.ghostUpdate(addv=PETSc.InsertMode.ADD, mode=PETSc.ScatterMode.REVERSE)
    fem.petsc.set_bc(b, bcs)

    # Solve linear system
    solver = PETSc.KSP().create(A.getComm())
    solver.setType("bcgs")
    solver.getPC().setType("bjacobi")
    solver.rtol = 1.0e-05
    solver.setOperators(A)
    solver.solve(b, target_func.x.petsc_vec)
    assert solver.reason > 0
    target_func.x.scatter_forward()

    # Destroy PETSc linear algebra objects and solver
    solver.destroy()
    A.destroy()
    b.destroy()

    return target_func
