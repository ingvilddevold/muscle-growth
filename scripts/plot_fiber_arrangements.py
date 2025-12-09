import pyvista as pv
import numpy as np
from pathlib import Path
import musclex
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from scipy.spatial.transform import Rotation as R

# --- Configuration ---
mesh_root_directory = Path(__file__).parents[1] / "meshes"

# Where to save the images
output_directory = (
    Path(__file__).parents[1] / "results" / "geometries" / "Panel_C_Fiber_Plots"
)
output_directory.mkdir(parents=True, exist_ok=True)

# 1. Colors
colors = {"TA": "#B22020", "BFLH": "#FF2752", "ST": "#FF7429"}

# 2. Camera View Configuration
# zooming in to reduce white space
views = {
    "TA": lambda p: (p.view_yz(), p.camera.zoom(1.8)),
    "BFLH": lambda p: (p.view_xz(), p.camera.zoom(1.8)),
    "ST": lambda p: (p.view_yz(), p.camera.zoom(1.8)),
}


def get_muscle_code(filename):
    if "TibialisAnterior" in filename:
        return "TA"
    if "BicepsFemoris" in filename:
        return "BFLH"
    if "Semi" in filename:
        return "ST"
    return None


def align_to_up_vector(points, target_axis=[0, 0, 1]):
    """
    Rotates points so their 'long axis' aligns with the target_axis (e.g., Z).
    """
    # 1. Centering: Subtract the mean so PCA works around the center of mass
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid

    # 2. PCA: Compute Covariance Matrix and Eigenvectors
    # The eigenvector corresponding to the largest eigenvalue is the "Long Axis"
    cov_matrix = np.cov(centered_points.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # Sort eigenvals/vecs (largest to smallest)
    sort_indices = np.argsort(eigenvalues)[::-1]
    principal_axis = eigenvectors[:, sort_indices[0]]

    # 3. Calculate Rotation Matrix
    # We want to rotate 'principal_axis' to match 'target_axis'
    target_axis = np.array(target_axis) / np.linalg.norm(target_axis)

    # Cross product gives the axis of rotation
    rot_axis = np.cross(principal_axis, target_axis)
    sin_angle = np.linalg.norm(rot_axis)
    cos_angle = np.dot(principal_axis, target_axis)

    # Handle the case where they are already aligned or opposite
    if sin_angle < 1e-6:
        # If already aligned, return Identity or Flip
        return np.eye(3) if cos_angle > 0 else -np.eye(3)

    rot_axis /= sin_angle  # Normalize rotation axis
    angle = np.arctan2(sin_angle, cos_angle)

    # Create rotation object
    r = R.from_rotvec(rot_axis * angle)
    rotation_matrix = r.as_matrix()

    return rotation_matrix, centroid


def process_meshes(root_dir):
    print(f"Scanning {root_dir} for .xdmf files...")

    # Set PyVista to off-screen mode for batch processing
    pv.OFF_SCREEN = True

    # Recursively find all .xdmf files in the mesh_root_directory
    for xdmf_path in root_dir.rglob("*.xdmf"):

        # 1. Construct expected fiber file path
        # Assumes naming convention: Name.xdmf -> Name_fibers.bp
        fiber_path = xdmf_path.parent / f"{xdmf_path.stem}_fibers.bp"

        if not fiber_path.exists():
            print(f"Skipping {xdmf_path.name}: Fiber file not found.")
            continue

        muscle_code = get_muscle_code(xdmf_path.name)
        if not muscle_code:
            continue

        print(f"Processing: {xdmf_path.stem}...")

        try:
            # 2. Load Geometry using MuscleX
            geometry = musclex.geometry.RealisticGeometry(
                str(xdmf_path), str(fiber_path)
            )

            # 3. Generate Plot
            plotter = geometry.plot(mode="fibers", glyph_subsampling=3)

            # 4. Apply Custom Color (Override MuscleX defaults)
            # We iterate through actors to find the fibers and recolor them
            target_color = colors[muscle_code]
            for actor in plotter.renderer.actors.values():
                if isinstance(actor, pv.Actor):
                    # Set the color of the fibers
                    actor.prop.color = target_color
                    # Optional: Add specular highlight for 3D effect
                    actor.prop.specular = 0.5
                    actor.prop.specular_power = 20

            # 5. Apply Camera View
            plotter.set_background("white")

            if muscle_code in views:
                views[muscle_code](plotter)
            else:
                plotter.view_iso()

            # 6. Save Screenshot
            output_filename = (
                output_directory / f"{xdmf_path.stem}_fiber_visualization.png"
            )
            plotter.screenshot(
                output_filename, scale=4
            )  # scale=4 for high-res publication quality

            plotter.close()
            print(f"Saved: {output_filename}")

        except Exception as e:
            print(f"Failed to process {xdmf_path.name}: {e}")


def crop_image_to_content(img_array, tolerance=0.9):
    """
    Auto-crops a white-background image to the bounding box of the non-white content.
    """
    # 1. Convert to grayscale/binary mask for detection
    # If RGBA, check alpha or RGB. If RGB, check if not white.
    if img_array.shape[2] == 4:  # RGBA
        # Check where Alpha > 0 OR RGB is not white
        mask = (img_array[:, :, 3] > 0) & (
            np.mean(img_array[:, :, :3], axis=2) < tolerance
        )
    else:  # RGB
        # Check where it's not white
        mask = np.mean(img_array, axis=2) < tolerance

    # 2. Find coordinates of pixels
    coords = np.argwhere(mask)

    if coords.size == 0:
        return img_array  # Return original if empty

    # 3. Get Bounding Box
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # 4. Slice (Add a tiny padding)
    pad = 100
    y_min = max(0, y_min - pad)
    x_min = max(0, x_min - pad)
    y_max = min(img_array.shape[0], y_max + pad)
    x_max = min(img_array.shape[1], x_max + pad)

    return img_array[y_min:y_max, x_min:x_max]


def assemble_wide_panel_c():
    # --- Configuration ---
    image_dir = output_directory

    # Structure: 1 Row of 3 "Muscle Groups"
    # Inside each group: 4 "Subjects"
    muscle_groups = [
        ("Biceps Femoris LH", "Biceps"),
        ("Semitendinosus", "Semitend"),
        ("Tibialis Anterior", "Tibialis"),
    ]

    subjects = [
        ("Female Left", "VHF_Left"),
        ("Female Right", "VHF_Right"),
        ("Male Left", "VHM_Left"),
        ("Male Right", "VHM_Right"),
    ]

    def find_file(directory, subject_key, muscle_key):
        all_files = list(directory.glob("*.png"))
        for f in all_files:
            if subject_key in f.name and muscle_key in f.name:
                return f
        return None

    # --- Plotting ---
    # We use a GridSpec to create 3 containers (one per muscle type)
    # Then sub-gridspecs for the 4 subjects inside each.

    # Figure Size: Wide and Short (e.g., 18 inches wide, 5 inches tall)
    fig = plt.figure(figsize=(7, 3), facecolor="white")

    # Main Grid: 1 Row, 3 Columns (for the 3 Muscles)
    outer_grid = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1], wspace=0.01)

    for group_idx, (muscle_label, muscle_key) in enumerate(muscle_groups):

        # Create a sub-grid for this muscle (1 Row, 4 Columns for subjects)
        inner_grid = gridspec.GridSpecFromSubplotSpec(
            1, 4, subplot_spec=outer_grid[group_idx], wspace=0.0
        )

        # Add a "Super Title" for the Muscle Group
        # We add a hidden axes to place the title centered over the block
        # title_ax = fig.add_subplot(outer_grid[group_idx], frameon=False)
        # title_ax.set_xticks([])
        # title_ax.set_yticks([])
        # title_ax.set_title(muscle_label, fontsize=14, fontweight='bold', pad=20)

        for subj_idx, (subj_label, subj_key) in enumerate(subjects):

            ax = fig.add_subplot(inner_grid[0, subj_idx])

            # Find and Load
            filepath = find_file(image_dir, subj_key, muscle_key)

            if filepath and filepath.exists():
                try:
                    img = mpimg.imread(filepath)

                    # --- AUTO CROP ---
                    img_cropped = crop_image_to_content(img)

                    ax.imshow(img_cropped)
                except Exception as e:
                    ax.text(0.5, 0.5, "Error", ha="center", color="red")
            else:
                ax.text(0.5, 0.5, "Missing", ha="center", color="red", fontsize=8)
                ax.set_facecolor("#f0f0f0")

            ax.axis("off")

            # Subject Labels (Bottom of the plot)
            short_labels = ["F-L", "F-R", "M-L", "M-R"]
            ax.text(
                0.5,
                -0.25,
                short_labels[subj_idx],
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=9,
            )

    # Save
    output_path = output_directory / "combined_fiber_arrangements.png"
    plt.savefig(output_path, dpi=500, bbox_inches="tight")
    print(f"Saved combined plot to: {output_path}")
    # plt.show()


if __name__ == "__main__":
    process_meshes(mesh_root_directory)
    assemble_wide_panel_c()
