from mpi4py import MPI
from dolfinx import mesh, geometry, cpp
import basix.ufl
import ufl
import numpy as np

def mark_cells(msh, cell_index):
    """
    Create a meshtag for a subset of cells.
    """
    num_cells = msh.topology.index_map(
        msh.topology.dim).size_local + msh.topology.index_map(
        msh.topology.dim).num_ghosts
    cells = np.arange(0, num_cells, dtype=np.int32)
    values = np.zeros_like(cells, dtype=np.int32)
    if len(cell_index) > 0:
        values[cell_index] = 1
    cell_tag = mesh.meshtags(msh, msh.topology.dim, cells, values)
    return cell_tag

def find_colliding_cells(mesh_big: mesh.Mesh, mesh_small: mesh.Mesh):
    """
    Finds overlapping/colliding cells between two meshes.

    Args:
        mesh_big: The first mesh.
        mesh_small: The second mesh.

    Returns:
        A tuple containing two dolfinx.mesh.MeshTags:
        - meshtags for colliding cells on the big mesh.
        - meshtags for colliding cells on the small mesh.
    """

    # Get number of cells for each mesh
    num_big_cells = mesh_big.topology.index_map(mesh_big.topology.dim).size_local + \
        mesh_big.topology.index_map(mesh_big.topology.dim).num_ghosts
    num_small_cells = mesh_small.topology.index_map(mesh_small.topology.dim).size_local + \
        mesh_small.topology.index_map(mesh_small.topology.dim).num_ghosts

    bb_tree = geometry.bb_tree(
        mesh_big, mesh_big.topology.dim, entities=np.arange(num_big_cells, dtype=np.int32))
    big_process = bb_tree.create_global_tree(mesh_big.comm)
    bb_small = geometry.bb_tree(
        mesh_small, mesh_small.topology.dim, entities=np.arange(num_small_cells, dtype=np.int32))

    process_collisions = geometry.compute_collisions_trees(
        bb_small, big_process)
    outgoing_edges = set()
    num_outgoing_cells = np.zeros(mesh_big.comm.size, dtype=np.int32)
    cell_indices = []
    for cell_idx, process_idx in process_collisions:
        num_outgoing_cells[process_idx] += 1
        outgoing_edges = set.union(outgoing_edges, (process_idx,))
    outgoing_edges = np.asarray(np.unique(list(outgoing_edges)), dtype=np.int32)
    small_to_big_comm = mesh_small.comm.Create_dist_graph(
        list([mesh_small.comm.rank]), [len(np.unique(outgoing_edges))], outgoing_edges, reorder=False)

    num_cells = num_small_cells
    source, dest, _ = small_to_big_comm.Get_dist_neighbors()

    num_vertices_per_cell_small = cpp.mesh.cell_num_vertices(
        mesh_small.topology.cell_type)

    # Extract all mesh nodes per process
    process_offsets = np.zeros(len(dest)+1, dtype=np.int32)
    np.cumsum(num_outgoing_cells[dest], out=process_offsets[1:])
    sending_cells = np.full(process_offsets[-1], 10, dtype=np.int32)
    insert_counter = np.zeros_like(dest, dtype=np.int32)
    for cell_idx, process_idx in process_collisions:
        local_idx = np.flatnonzero(dest == process_idx)
        assert len(local_idx) == 1
        idx = local_idx[0]
        sending_cells[process_offsets[idx]+insert_counter[idx]] = cell_idx
        insert_counter[idx] += 1


    node_counter = np.zeros(mesh_small.geometry.index_map().size_local +
                            mesh_small.geometry.index_map().num_ghosts+1, dtype=np.int32)
    local_pos = np.zeros_like(node_counter, dtype=np.int32)
    send_geom = []
    send_top = []
    send_top_size = np.zeros_like(dest, dtype=np.int32)
    send_geom_size = np.zeros_like(dest, dtype=np.int32)

    for i in range(len(dest)):
        # Get nodes of all cells sent to a given process
        org_nodes = cpp.mesh.entities_to_geometry(
            mesh_small._cpp_object, mesh_small.topology.dim, sending_cells[process_offsets[i]:process_offsets[i+1]], False).reshape(-1)

        # Get the unique set of nodes sent to this process
        unique_nodes = np.unique(org_nodes)

        # Compute remapping of nodes
        node_counter[:] = 0
        node_counter[unique_nodes] = 1
        np.cumsum(node_counter, out=local_pos)
        local_pos -= 1  # Map to 0 index system
        send_geom.append(
            mesh_small.geometry.x[unique_nodes][:, :mesh_small.geometry.dim])
        send_geom_size[i] = np.size(send_geom[-1])
        send_top.append(local_pos[org_nodes])
        send_top_size[i] = np.size(send_top[-1])

    # Compute send and receive offsets for geometry and topology
    geom_offset = np.zeros(len(dest)+1, dtype=np.int32)
    top_offset = np.zeros(len(dest)+1, dtype=np.int32)
    np.cumsum(send_geom_size, out=geom_offset[1:])
    np.cumsum(send_top_size, out=top_offset[1:])

    if len(send_geom) == 0:
        send_geom = np.array([], dtype=mesh_small.geometry.x.dtype)
    else:
        send_geom = np.vstack(send_geom).reshape(-1)
    if len(send_top) == 0:
        send_top = np.array([], dtype=np.int32)
    else:
        send_top = np.hstack(send_top)
    if len(send_geom_size) == 0:
        send_geom_size = np.zeros(1, dtype=np.int32)
    if len(send_top_size) == 0:
        send_top_size = np.zeros(1, dtype=np.int32)

    recv_geom_size = np.zeros(max(len(source), 1), dtype=np.int32)
    small_to_big_comm.Neighbor_alltoall(send_geom_size, recv_geom_size)
    recv_geom_size = recv_geom_size[:len(source)]
    send_geom_size = send_geom_size[:len(dest)]
    recv_geom_offsets = np.zeros(len(recv_geom_size)+1, dtype=np.int32)
    np.cumsum(recv_geom_size, out=recv_geom_offsets[1:])


    recv_top_size = np.zeros(max(len(source), 1), dtype=np.int32)
    small_to_big_comm.Neighbor_alltoall(send_top_size, recv_top_size)
    recv_top_size = recv_top_size[:len(source)]
    send_top_size = send_top_size[:len(dest)]

    recv_top_offsets = np.zeros(len(recv_top_size)+1, dtype=np.int32)
    np.cumsum(recv_top_size, out=recv_top_offsets[1:])

    numpy_to_mpi = {np.float64: MPI.DOUBLE,
                    np.float32: MPI.FLOAT, np.int8: MPI.INT8_T, np.int32: MPI.INT32_T}
    # Communicate data
    recv_geom = np.zeros(recv_geom_offsets[-1], dtype=mesh_small.geometry.x.dtype)
    s_geom_msg = [send_geom, send_geom_size, numpy_to_mpi[send_geom.dtype.type]]
    r_geom_msg = [recv_geom, recv_geom_size, numpy_to_mpi[recv_geom.dtype.type]]
    small_to_big_comm.Neighbor_alltoallv(s_geom_msg, r_geom_msg)

    recv_top = np.zeros(recv_top_offsets[-1], dtype=np.int32)
    s_top_msg = [send_top, send_top_size, numpy_to_mpi[send_top.dtype.type]]
    r_top_msg = [recv_top, recv_top_size, numpy_to_mpi[recv_top.dtype.type]]
    small_to_big_comm.Neighbor_alltoallv(s_top_msg, r_top_msg)

    # For each received geometry, create a mesh
    local_meshes = []
    for i in range(len(source)):
        local_meshes.append(mesh.create_mesh(
            MPI.COMM_SELF,
            cells=recv_top[recv_top_offsets[i]:recv_top_offsets[i+1]].reshape(-1, num_vertices_per_cell_small),
            x=recv_geom[recv_geom_offsets[i]:recv_geom_offsets[i+1]].reshape(-1, mesh_small.geometry.dim),
            e=ufl.Mesh(mesh_small.ufl_domain().ufl_coordinate_element())))


    def extract_cell_geometry(input_mesh, cell: int):
        mesh_nodes = cpp.mesh.entities_to_geometry(
            input_mesh._cpp_object, input_mesh.topology.dim, np.array([cell], dtype=np.int32), False)[0]
        
        return input_mesh.geometry.x[mesh_nodes]


    # For each local mesh, compute the bounding box, compute colliding cells
    tol = 1e-13
    big_cells = []
    local_cells = []
    num_local_cells = np.zeros(max(len(source), 1), dtype=np.int32)
    for i in range(len(source)):
        local_cells_i = set()
        o_cell_idx = local_meshes[i].topology.original_cell_index
        local_tree = geometry.bb_tree(
            local_meshes[i], local_meshes[i].topology.dim)
        cell_cell_collisions = geometry.compute_collisions_trees(
            local_tree, bb_tree)
        for local_cell, big_cell in cell_cell_collisions:

            geom_small = extract_cell_geometry(local_meshes[i], local_cell)
            geom_big = extract_cell_geometry(mesh_big, big_cell)
            distance = geometry.compute_distance_gjk(geom_big, geom_small)
            if np.linalg.norm(distance) <= tol:
                big_cells.append(big_cell)
                local_cells_i = local_cells_i.union([o_cell_idx[local_cell]])
        num_local_cells[i] = len(local_cells_i)
        local_cells.append(np.asarray(list(local_cells_i), dtype=np.int32))


    # Create reverse communicator
    big_to_small_comm = mesh_small.comm.Create_dist_graph_adjacent(
        dest, source, reorder=False
    )
    # Send incoming cell sizes
    recv_cell_sizes = np.zeros(max(len(dest), 1), dtype=np.int32)
    big_to_small_comm.Neighbor_alltoall(num_local_cells, recv_cell_sizes)
    recv_cell_sizes = recv_cell_sizes[:len(dest)]
    num_local_cells = num_local_cells[:len(source)]

    recv_cell_offsets = np.zeros(len(recv_cell_sizes)+1, dtype=np.int32)
    np.cumsum(recv_cell_sizes, out=recv_cell_offsets[1:])
    recv_cells = np.zeros(recv_cell_offsets[-1], np.int32)
    if len(source) == 0:
        send_cells = np.array([], dtype=np.int32)
    else:
        send_cells = np.hstack(local_cells).astype(np.int32).reshape(-1)
    s_cell_msg = [send_cells, num_local_cells, numpy_to_mpi[send_cells.dtype.type]]
    r_cell_msg = [recv_cells, recv_cell_sizes, numpy_to_mpi[recv_cells.dtype.type]]
    big_to_small_comm.Neighbor_alltoallv(s_cell_msg, r_cell_msg)

    small_mesh_colliding_cells = []
    for i in range(len(dest)):
        local_cells = sending_cells[process_offsets[i]:process_offsets[i+1]]
        recv_data = recv_cells[recv_cell_offsets[i]:recv_cell_offsets[i+1]]
        small_mesh_colliding_cells.append(local_cells[recv_data])
    if len(small_mesh_colliding_cells) == 0:
        small_mesh_colliding_cells = np.array([], dtype=np.int32)
    else:
        small_mesh_colliding_cells = np.hstack(small_mesh_colliding_cells)

    sorted_unique_small_cells = np.unique(
        small_mesh_colliding_cells).astype(dtype=np.int32)
    colliding_small_marker = mark_cells(mesh_small, sorted_unique_small_cells)
    colliding_small_marker.name = "colliding cells small"


    sorted_unique_big_cells = np.unique(big_cells).astype(dtype=np.int32)
    colliding_big_marker = mark_cells(mesh_big, sorted_unique_big_cells)
    colliding_big_marker.name = "colliding cells"

    return colliding_big_marker, colliding_small_marker


def create_midpoint_plane(mesh_3d: mesh.Mesh, padding: float = 1.2, resolution: int = 20):
    """
    Creates a rectangular mesh that intersects the midpoint of the muscle
    along its principal (long) axis.

    Args:
        padding: Factor to scale the plane size relative to the mesh's max dimension.
        resolution: Number of cells along each axis of the plane mesh.

    Returns:
        A 2D dolfinx.mesh.Mesh representing the intersecting plane, embedded in 3D.
    """

    comm = mesh_3d.comm
    xtype = mesh_3d.geometry.x.dtype

    # 1. Get points and perform PCA to find the principal axes
    points = mesh_3d.geometry.x
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    cov_matrix = np.cov(centered_points, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvectors by eigenvalues (largest first)
    sort_indices = np.argsort(eigenvalues)[::-1]
    major_axis = eigenvectors[:, sort_indices[0]] # Long axis
    axis_2 = eigenvectors[:, sort_indices[1]]
    axis_3 = eigenvectors[:, sort_indices[2]]

    # 2. Find the muscle's extent along its major axis to locate the midpoint
    projections = centered_points @ major_axis
    min_proj, max_proj = np.min(projections), np.max(projections)
    plane_origin = centroid + ((min_proj + max_proj) / 2) * major_axis

    # 3. Define the rectangle size based on the muscle's extent along the
    #    other two principal axes (the cross-section dimensions). This creates
    #    a much tighter bounding box than using the overall max dimension.
    size_2 = np.ptp(centered_points @ axis_2) * padding
    size_3 = np.ptp(centered_points @ axis_3) * padding

    # 4. Create a 2D template mesh and warp it into the 3D plane
    rect_mesh_2d = mesh.create_rectangle(
        comm, 
        [np.array([-size_2/2, -size_3/2]), np.array([size_2/2, size_3/2])],
        [resolution, resolution],
        dtype=xtype
    )

    x_3d = np.zeros((rect_mesh_2d.geometry.x.shape[0], 3), dtype=xtype)
    # Map the 2D mesh points into the 3D plane defined by the PCA axes and origin
    for i, p_2d in enumerate(rect_mesh_2d.geometry.x):
        x_3d[i, :] = plane_origin + p_2d[0] * axis_2 + p_2d[1] * axis_3

    topology = rect_mesh_2d.topology.connectivity(2, 0).array.reshape(-1, 3)
    org_coordinate_element = rect_mesh_2d.ufl_domain().ufl_coordinate_element()
    coordinate_element = ufl.Mesh(basix.ufl.element(org_coordinate_element.family_name, rect_mesh_2d.basix_cell(), org_coordinate_element.degree, shape=(3,)))
    plane_mesh = mesh.create_mesh(comm, cells=topology, x=x_3d, e=coordinate_element)
    return plane_mesh, major_axis


def create_midpoint_box(mesh_3d: mesh.Mesh, thickness: float, padding: float = 1.2, resolution_xyz=(5, 50, 50)):
    """
    Creates a 3D thin box mesh that intersects the midpoint of a 3D mesh
    along its principal (long) axis, and tags one of its faces.

    Args:
        mesh_3d: The 3D dolfinx.mesh.Mesh to intersect.
        thickness: The thickness of the box along the principal axis.
        padding: Factor to scale the box's cross-section size.
        resolution_xyz: Number of cells for the box along its local x, y, z axes.

    Returns:
        A tuple (box_mesh, box_facet_tags).
    """
    # 1. Perform PCA on the 3D mesh to find centroid and orientation
    coords = mesh_3d.geometry.x
    centroid = np.mean(coords, axis=0)
    coords_centered = coords - centroid
    U, s, Vt = np.linalg.svd(coords_centered)
    principal_axes = Vt.T # Columns are the principal axes

    # 2. Project points onto the plane perpendicular to the main axis to find bounds
    main_axis = principal_axes[:, 0]
    plane_axis_1 = principal_axes[:, 1]
    plane_axis_2 = principal_axes[:, 2]
    
    projected_coords_1 = coords_centered @ plane_axis_1
    projected_coords_2 = coords_centered @ plane_axis_2
    
    min_1, max_1 = np.min(projected_coords_1), np.max(projected_coords_1)
    min_2, max_2 = np.min(projected_coords_2), np.max(projected_coords_2)
    
    # 3. Define local box dimensions
    half_dim_1 = (max_1 - min_1) / 2 * padding
    half_dim_2 = (max_2 - min_2) / 2 * padding
    half_thickness = thickness / 2
    
    p0 = [-half_thickness, -half_dim_1, -half_dim_2]
    p1 = [half_thickness, half_dim_1, half_dim_2]

    # 4. Create an axis-aligned box and its facet tags
    box_local = mesh.create_box(MPI.COMM_WORLD, [p0, p1], resolution_xyz, mesh.CellType.tetrahedron)
    
    # Tag the face at x = -half_thickness
    def bottom_face_local(x):
        return np.isclose(x[0], -half_thickness)
    
    bottom_facets = mesh.locate_entities_boundary(box_local, box_local.topology.dim - 1, bottom_face_local)
    bottom_values = np.full_like(bottom_facets, 1, dtype=np.int32)
    sorted_facets = np.argsort(bottom_facets)
    box_facet_tags = mesh.meshtags(box_local, box_local.topology.dim - 1, bottom_facets[sorted_facets], bottom_values[sorted_facets])

    # 5. Rotate and translate the box to align with the muscle
    # The rotation matrix aligns the local box's x-axis with the muscle's main axis
    rotation_matrix = principal_axes
    box_local.geometry.x[:] = (rotation_matrix @ box_local.geometry.x.T).T + centroid

    return box_local, box_facet_tags
