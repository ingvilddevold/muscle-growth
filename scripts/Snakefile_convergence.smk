# Simulations with mesh and time refinement for verification
#
# Run script as:
#   $ snakemake -s scripts/Snakefile_convergence.smk --profile scripts/ex3

from pathlib import Path

# --- Global Configuration ---
ROOT_DIR = Path(workflow.basedir).parent
RESULTS_DIR = ROOT_DIR / "results"
CONFIG_DIR = ROOT_DIR / "config_files"
SCRIPT_DIR = Path(workflow.basedir)
SRC_DIR = Path(workflow.basedir).parent / "src" / "musclex"

# Path for outputs
PAPER_FIG_DIR = RESULTS_DIR / "convergence"
PAPER_FIG_DIR.mkdir(exist_ok=True, parents=True)

# --- Define Simulations ---
# Simulating a short one-week protocol with daily exercise
PROTOCOL = "oneweek"

MESHES = {
    "idealizedRef0": ROOT_DIR / "meshes" / "muscle-idealized",
    "idealizedRef1": ROOT_DIR / "meshes" / "muscle-idealizedRef1",
    "idealizedRef2": ROOT_DIR / "meshes" / "muscle-idealizedRef2",
}

CASES = [
    {"mesh": "idealizedRef0", "dt": 1.0},
    {"mesh": "idealizedRef0", "dt": 0.5},
    {"mesh": "idealizedRef1", "dt": 1.0},
    {"mesh": "idealizedRef2", "dt": 1.0},
]

RESOURCES = {
    "idealizedRef0": {"ntasks": 8, "time": "00:30:00"},
    "idealizedRef1": {"ntasks": 32, "time": "01:00:00"},
    "idealizedRef2": {"ntasks": 64, "time": "06:00:00"},
}

MESH_DISPLAY_NAMES = {
    "idealizedRef0": "Coarse mesh",
    "idealizedRef1": "Medium mesh",
    "idealizedRef2": "Fine mesh"
}
# Output example: "Coarse mesh, dt=1.0h"
LEGEND_LABELS = [f"{MESH_DISPLAY_NAMES[c['mesh']]}, dt={c['dt']}h" for c in CASES]

# Replaces `expand` to strictly enforce the specific mesh/dt pairs defined in CASES
TARGET_CSV_FILES = [
    PAPER_FIG_DIR / f"{c['mesh']}_{c['dt']}" / f"growth_sim_{PROTOCOL}" / "growth_results.csv"
    for c in CASES
]

rule all:
    input:
        PAPER_FIG_DIR / "mesh_convergence.png"


rule runGrowthSimulation:
    output:
        sim_dir=directory(PAPER_FIG_DIR / "{mesh}_{dt}" / "growth_sim_{protocol}"),
        results_csv=PAPER_FIG_DIR / "{mesh}_{dt}" / "growth_sim_{protocol}" / "growth_results.csv",
    input:
        exercise_config=CONFIG_DIR / "exercise_eq_reduced_k1.yml",
        material_config=CONFIG_DIR / "material_rohrle.yml",
    params:
        mesh_name="{mesh}",
        mesh_path=lambda wildcards: MESHES[wildcards.mesh],
        protocol_name="{protocol}",
    wildcard_constraints:
        protocol=PROTOCOL,
    conda:
        "musclex2"
    resources:
        time=lambda wildcards: RESOURCES[wildcards.mesh]["time"],
        ntasks=lambda wildcards: RESOURCES[wildcards.mesh]["ntasks"],
    shell:
        """
        export OMP_NUM_THREADS=1 && \
        mpirun -n {resources.ntasks} python {SCRIPT_DIR}/run_coupled_growth.py \
            --exercise-config {input.exercise_config} \
            --material-config {input.material_config} \
            --mesh-path {params.mesh_path} \
            --protocol-name {params.protocol_name} \
            --output-dir {output.sim_dir} \
            --dt-growth {wildcards.dt}
        """

rule plotMeshConvergence:
    localrule: True
    input:
        csvs=TARGET_CSV_FILES,
        script=SCRIPT_DIR / "plot_mesh_convergence.py",
    output:
        PAPER_FIG_DIR / "mesh_convergence.png",
    params:
        labels="|".join(LEGEND_LABELS), 
    conda:
        "musclex2"
    shell:
        """
        python {input.script} \
            {input.csvs} \
            --labels "{params.labels}" \
            --output-file {output}
        """
