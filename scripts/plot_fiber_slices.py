from pathlib import Path

import adios4dolfinx
import dolfinx
import dolfinx.plot
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from mpi4py import MPI


# ------- Helper functions -------
def crop_white_borders(img_array):
    """Crops all pure white background pixels from an image array."""
    non_white_mask = np.any(img_array[..., :3] < 255, axis=-1)
    rows = np.any(non_white_mask, axis=1)
    cols = np.any(non_white_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return img_array
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    pad = 5
    return img_array[
        max(0, y_min - pad) : min(img_array.shape[0], y_max + pad),
        max(0, x_min - pad) : min(img_array.shape[1], x_max + pad),
    ]


def render_muscle_images(meshfile):
    """Loads FEniCSx data, slices it, and renders cropped PyVista arrays."""
    with dolfinx.io.XDMFFile(MPI.COMM_WORLD, meshfile, "r") as f:
        domain = f.read_mesh(name="Grid")
        domain.topology.create_connectivity(
            domain.topology.dim - 1, domain.topology.dim
        )

    domain.geometry.x[:] = domain.geometry.x * 1e-3
    V_fibers = dolfinx.fem.functionspace(
        domain, ("Lagrange", 2, (domain.geometry.dim,))
    )
    fibers = dolfinx.fem.Function(V_fibers, name="fiberdirection")

    fibersfile = meshfile.parent / f"{meshfile.stem}_fibers.bp"
    adios4dolfinx.read_function(fibersfile, fibers)

    V_vis = dolfinx.fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
    fibers_vis = dolfinx.fem.Function(V_vis)
    fibers_vis.interpolate(fibers)

    topology, cell_types, geometry = dolfinx.plot.vtk_mesh(V_vis)
    grid = pv.UnstructuredGrid(topology, cell_types, geometry)
    grid.point_data["Fiber_Direction"] = fibers_vis.x.array.reshape(-1, 3)

    if grid.point_data["Fiber_Direction"][:, 2].mean() < 0:
        grid.point_data["Fiber_Direction"] *= -1

    grid.point_data["Fiber_X"] = grid.point_data["Fiber_Direction"][:, 0]
    grid.point_data["Fiber_Y"] = grid.point_data["Fiber_Direction"][:, 1]
    grid.point_data["Fiber_Z"] = grid.point_data["Fiber_Direction"][:, 2]

    # Calculate 2 Slices (Proximal first, then Distal)
    z_min, z_max = grid.bounds[4], grid.bounds[5]
    z_length = z_max - z_min

    # Higher Z is Proximal, Lower Z is Distal.
    slice_prox = grid.slice(normal=[0, 0, 1], origin=[0, 0, z_min + 0.66 * z_length])
    slice_dist = grid.slice(normal=[0, 0, 1], origin=[0, 0, z_min + 0.33 * z_length])

    slices = [slice_prox, slice_dist]

    components = ["Fiber_X", "Fiber_Y", "Fiber_Z"]
    clims = []
    for comp in components:
        c_min, c_max = grid.point_data[comp].min(), grid.point_data[comp].max()
        if np.isclose(c_min, c_max):
            c_min -= 0.01
            c_max += 0.01
        clims.append([c_min, c_max])

    pl_loc = pv.Plotter(window_size=(1000, 1600), off_screen=True)
    pl_loc.set_background("white")
    pl_loc.add_mesh(grid, color="whitesmoke", opacity=0.15, show_edges=False)
    pl_loc.add_mesh(slices[0], color="indianred", opacity=0.9)
    pl_loc.add_mesh(slices[1], color="steelblue", opacity=0.9)
    pl_loc.camera_position = "iso"
    pl_loc.add_axes(
        color="black", viewport=(0.2, 0.2, 0.4, 0.4), line_width=3, labels_off=False
    )
    img_locator = crop_white_borders(pl_loc.screenshot(return_img=True))
    pl_loc.close()

    # Find Zoom
    # Find the maximum X or Y extent of the entire muscle to preserve relative sizing
    grid_dx = grid.bounds[1] - grid.bounds[0]
    grid_dy = grid.bounds[3] - grid.bounds[2]

    # parallel_scale represents exactly half the height of the camera viewport.
    # We take the maximum dimension, divide by 2, and multiply by 1.05 for a 5% padding.
    optimal_scale = max(grid_dx, grid_dy) / 2.0 * 1.05

    # Render Slices
    slice_imgs = {}
    for comp, clim in zip(components, clims):
        for i, slc in enumerate(slices):
            pl = pv.Plotter(window_size=(800, 800), off_screen=True)
            pl.set_background("white")
            pl.add_mesh(
                slc, scalars=comp, cmap="viridis", clim=clim, show_scalar_bar=False
            )

            pl.camera_position = "xy"
            pl.enable_parallel_projection()

            # Look directly down the Z-axis, centered precisely on this specific slice
            pl.camera.focal_point = slc.center
            pl.camera.position = (slc.center[0], slc.center[1], slc.center[2] + 1.0)
            pl.camera.parallel_scale = optimal_scale

            slice_imgs[f"{comp}_{i}"] = crop_white_borders(
                pl.screenshot(return_img=True)
            )
            pl.close()

    return img_locator, slice_imgs, clims


# ------- Define paths to meshes -------
meshes_dir = Path(__file__).parents[1] / "meshes"

all_muscle_groups = {
    "Biceps_Femoris_Long_Head": [
        {
            "name": "Male Left",
            "mesh": meshes_dir
            / "VHM_Left_Muscle_BicepsFemorisLongus_smooth/VHM_Left_Muscle_BicepsFemorisLongus_smooth.xdmf",
        },
        {
            "name": "Male Right",
            "mesh": meshes_dir
            / "VHM_Right_Muscle_BicepsFemorisLong_smooth/VHM_Right_Muscle_BicepsFemorisLong_smooth.xdmf",
        },
        {
            "name": "Female Left",
            "mesh": meshes_dir
            / "VHF_Left_Muscle_BicepsFemorisLong_smooth/VHF_Left_Muscle_BicepsFemorisLong_smooth.xdmf",
        },
        {
            "name": "Female Right",
            "mesh": meshes_dir
            / "VHF_Right_Muscle_BicepsFemorisLongHead_smooth/VHF_Right_Muscle_BicepsFemorisLongHead_smooth.xdmf",
        },
    ],
    "Tibialis_Anterior": [
        {
            "name": "Male Left",
            "mesh": meshes_dir
            / "VHM_Left_Muscle_TibialisAnterior_smooth/VHM_Left_Muscle_TibialisAnterior_smooth.xdmf",
        },
        {
            "name": "Male Right",
            "mesh": meshes_dir
            / "VHM_Right_Muscle_TibialisAnterior_smooth/VHM_Right_Muscle_TibialisAnterior_smooth.xdmf",
        },
        {
            "name": "Female Left",
            "mesh": meshes_dir
            / "VHF_Left_Muscle_TibialisAnterior_smooth/VHF_Left_Muscle_TibialisAnterior_smooth.xdmf",
        },
        {
            "name": "Female Right",
            "mesh": meshes_dir
            / "VHF_Right_Muscle_TibialisAnterior_smooth/VHF_Right_Muscle_TibialisAnterior_smooth.xdmf",
        },
    ],
    "Semitendinosus": [
        {
            "name": "Male Left",
            "mesh": meshes_dir
            / "VHM_Left_Muscle_Semitendonosus_smooth/VHM_Left_Muscle_Semitendonosus_smooth.xdmf",
        },
        {
            "name": "Male Right",
            "mesh": meshes_dir
            / "VHM_Right_Muscle_Semitendinosus_smooth/VHM_Right_Muscle_Semitendinosus_smooth.xdmf",
        },
        {
            "name": "Female Left",
            "mesh": meshes_dir
            / "VHF_Left_Muscle_Semitendinosus_smooth/VHF_Left_Muscle_Semitendinosus_smooth.xdmf",
        },
        {
            "name": "Female Right",
            "mesh": meshes_dir
            / "VHF_Right_Muscle_Semitendonosus_smooth/VHF_Right_Muscle_Semitendonosus_smooth.xdmf",
        },
    ],
}


# ------- Make subplots -------
print("=== Rendering all 12 muscles ===")
master_data = []  # Will be a list of lists: master_data[type_idx][muscle_idx]
for muscle_type, variations in all_muscle_groups.items():
    formatted_type = muscle_type.replace("_", " ")
    type_data = []
    for var in variations:
        print(f"  Rendering {formatted_type} - {var['name']}...")
        loc, imgs, clims = render_muscle_images(var["mesh"])
        type_data.append(
            {
                "type": formatted_type,
                "name": var["name"],
                "locator": loc,
                "images": imgs,
                "clims": clims,
            }
        )
    master_data.append(type_data)

# ------- Build full figure -------
components = ["Fiber_X", "Fiber_Y", "Fiber_Z"]
col_titles = ["Proximal", "Distal"]
row_titles = ["X-Comp", "Y-Comp", "Z-Comp"]

width_ratios = [1.5, 1, 1, 0.15] * 3
num_cols = len(width_ratios)

layout = []
height_ratios = []

for m_idx in range(4):
    row_x, row_y, row_z = [], [], []
    for t_idx in range(3):
        row_x.extend(
            [
                f"loc_{t_idx}_{m_idx}",
                f"X_{t_idx}_{m_idx}_0",
                f"X_{t_idx}_{m_idx}_1",
                f"cbar_X_{t_idx}_{m_idx}",
            ]
        )
        row_y.extend(
            [
                f"loc_{t_idx}_{m_idx}",
                f"Y_{t_idx}_{m_idx}_0",
                f"Y_{t_idx}_{m_idx}_1",
                f"cbar_Y_{t_idx}_{m_idx}",
            ]
        )
        row_z.extend(
            [
                f"loc_{t_idx}_{m_idx}",
                f"Z_{t_idx}_{m_idx}_0",
                f"Z_{t_idx}_{m_idx}_1",
                f"cbar_Z_{t_idx}_{m_idx}",
            ]
        )

    layout.append(row_x)
    layout.append(row_y)
    layout.append(row_z)

    # Assign normal height (1) to the X, Y, and Z data rows
    height_ratios.extend([1, 1, 1])

    # Insert an empty "spacer" row between subjects (but not after the very last one)
    if m_idx < 3:
        layout.append(
            ["."] * num_cols
        )  # '.' tells Matplotlib to leave this grid cell empty
        height_ratios.append(
            0.5
        )  # 0.5 means the gap is half the height of a normal row

fig, axes = plt.subplot_mosaic(
    layout,
    figsize=(24, 23),
    gridspec_kw={
        "width_ratios": width_ratios,
        "height_ratios": height_ratios,
        "wspace": 0.1,
        "hspace": 0.05,
    },
)

for t_idx, type_data in enumerate(master_data):
    for m_idx, data in enumerate(type_data):
        # Plot 3D Context
        ax_loc = axes[f"loc_{t_idx}_{m_idx}"]
        ax_loc.imshow(data["locator"])
        ax_loc.axis("off")

        if m_idx == 0:
            ax_loc.set_title(
                f"{data['type']}\n{data['name']}",
                fontsize=16,
                fontweight="bold",
                pad=10,
            )
        else:
            ax_loc.set_title(f"{data['name']}", fontsize=14, fontweight="bold", pad=5)

        for row_idx, (comp, clim) in enumerate(zip(components, data["clims"])):
            for col_idx in [0, 1]:
                ax = axes[f"{comp[-1]}_{t_idx}_{m_idx}_{col_idx}"]
                ax.imshow(data["images"][f"{comp}_{col_idx}"])
                ax.axis("off")

                if m_idx == 0 and row_idx == 0:
                    ax.set_title(
                        col_titles[col_idx], fontsize=14, fontweight="bold", pad=10
                    )

                if col_idx == 0:
                    ax.text(
                        -0.1,
                        0.5,
                        row_titles[row_idx],
                        va="center",
                        ha="right",
                        rotation=90,
                        fontsize=12,
                        fontweight="bold",
                        transform=ax.transAxes,
                    )

            # Dedicated colorbar slot
            norm = mcolors.Normalize(vmin=clim[0], vmax=clim[1])
            sm = cm.ScalarMappable(cmap="viridis", norm=norm)
            sm.set_array([])

            cax = axes[f"cbar_{comp[-1]}_{t_idx}_{m_idx}"]
            cbar = fig.colorbar(sm, cax=cax)
            cbar.ax.tick_params(labelsize=9)

plt.subplots_adjust(left=0.03, right=0.97, top=0.95, bottom=0.02)
plt.savefig("muscle_fiber_slices.png", dpi=300, bbox_inches="tight", pad_inches=0.1)
print("Saved figure to: muscle_fiber_slices.png ")
