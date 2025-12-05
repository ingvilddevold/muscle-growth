"""
Visualize the 3D muscle geometries with tissue-specific coloring and custom highlights.
Assumes STL files are downloaded and stores in root_directory.
Will load and visualize all STL files found in the directory and its subdirectories.
Download from: https://digitalcommons.du.edu/visiblehuman/1/ (Final 3D STL Models)
"""

import pyvista as pv
import os


def visualize_anatomy(root_directory, custom_highlights=None):

    # --- COLOR PALETTE ---
    theme = {
        "bone": "#FFFFFF",
        "muscle_context": "#F8C4CF",
        "ligament": "#FFFFFF",
        "cartilage": "#FFFFFF",
        # Settings
        "context_opacity": 1,
        "background": "white",
    }

    TISSUE_COLORS = {
        "Bone": theme["bone"],
        "Muscle": theme["muscle_context"],
        "Ligament": theme["ligament"],
        "Cartilage": theme["cartilage"],
    }

    DEFAULT_COLOR = "lightgrey"

    if custom_highlights is None:
        custom_highlights = {}

    plotter = pv.Plotter(window_size=[800, 1600], off_screen=True)
    plotter.set_background(theme["background"])

    light = pv.Light(position=(10, 10, 10), show_actor=True, intensity=0.1)
    plotter.add_light(light)

    print(f"Scanning directory: {root_directory}")
    loaded_count = 0

    for subdir, dirs, files in os.walk(root_directory):
        for filename in files:
            if filename.lower().endswith(".stl"):
                filepath = os.path.join(subdir, filename)

                color = DEFAULT_COLOR
                opacity = 1.0

                # 1. Custom Highlights
                if filename in custom_highlights:
                    color = custom_highlights[filename]
                    opacity = 1.0
                    specular = 0.5  # Shinier highlights

                # 2. General Tissues
                elif "_Bone_" in filename:
                    color = TISSUE_COLORS["Bone"]
                    specular = 0.1  # Bones are matte
                elif "_Muscle_" in filename:
                    color = TISSUE_COLORS["Muscle"]
                    opacity = theme["context_opacity"]
                    specular = 0.2
                elif "_Ligament_" in filename:
                    color = TISSUE_COLORS["Ligament"]
                    specular = 0.3
                elif "_Cartilage_" in filename:
                    color = TISSUE_COLORS["Cartilage"]
                    opacity = 0.8
                    specular = 0.4

                try:
                    mesh = pv.read(filepath)
                    plotter.add_mesh(
                        mesh,
                        color=color,
                        opacity=opacity,
                        specular=specular,
                    )
                    loaded_count += 1
                except Exception as e:
                    print(f"Failed to load {filename}: {e}")

    print(f"Total parts loaded: {loaded_count}")

    # Define output filenames
    front_filename = "view_front.png"
    back_filename = "view_back.png"

    # --- A. Generate Front View ---
    print("Generating Front View...")
    plotter.view_xz(negative=True)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.zoom(1.8)
    # Force an update and save
    plotter.render()
    plotter.screenshot(front_filename)
    print(f"Saved: {front_filename}")

    # --- B. Generate Back View ---
    print("Generating Back View...")
    plotter.view_xz(negative=False)
    plotter.camera.up = (0, 0, 1)
    plotter.camera.zoom(1.8)
    # Force an update and save
    plotter.render()
    plotter.screenshot(back_filename)
    print(f"Saved: {back_filename}")

    # Clean up memory
    plotter.close()
    print("Process complete.")


if __name__ == "__main__":
    base_path = os.path.expanduser("~/Downloads/Final 3D STL Models-stl")

    # Define custom highlights for specific muscles
    color_ta = "#B22020"
    color_bflh = "#FF2752"
    color_st = "#FF7429"

    my_highlights = {
        "VHF_Left_Muscle_TibialisAnterior_smooth.stl": color_ta,
        "VHF_Right_Muscle_TibialisAnterior_smooth.stl": color_ta,
        "VHF_Left_Muscle_BicepsFemorisLong_smooth.stl": color_bflh,
        "VHF_Right_Muscle_BicepsFemorisLongHead_smooth.stl": color_bflh,
        "VHF_Left_Muscle_Semitendinosus_smooth.stl": color_st,
        "VHF_Right_Muscle_Semitendonosus_smooth.stl": color_st,
    }

    visualize_anatomy(base_path, custom_highlights=my_highlights)
