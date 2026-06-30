# Simulations and postprocessing for supplementary figure showing
# growth rate reduction due to feedback
#
# Run script as:
#   $ snakemake -s scripts/Snakefile_idealized_long.smk --use-conda --cores 4

from pathlib import Path

# --- Global Configuration ---
ROOT_DIR = Path(workflow.basedir).parent
RESULTS_DIR = ROOT_DIR / "results"
CONFIG_DIR = ROOT_DIR / "config_files"
SCRIPT_DIR = Path(workflow.basedir)
SRC_DIR = Path(workflow.basedir).parent / "src" / "musclex"
# Path for outputs
PAPER_FIG_DIR = RESULTS_DIR / "figure4"
PAPER_FIG_DIR.mkdir(exist_ok=True, parents=True)

# --- Define Simulations ---
# Simulating three exercise protocols on idealized fusiform mesh
PROTOCOLS = ["long"]
MESHES = {"idealized_fusiform": ROOT_DIR / "meshes" / "muscle-idealized"}

# --- Postprocessing Parameters ---
POSTPROCESS_FREQ = 10
POSTPROCESS_WARP_SCALE = 10.0


rule all:
    input:
        expand(
            PAPER_FIG_DIR / "{mesh}" / "growth_sim_long" / "growth_rate.png", mesh=MESHES.keys()
        ),


rule runGrowthSimulation:
    output:
        results_csv=PAPER_FIG_DIR
        / "{mesh}"
        / "growth_sim_long"
        / "growth_results.csv",
    input:
        exercise_config=CONFIG_DIR / "exercise_eq_reduced_k1.yml",
        material_config=CONFIG_DIR / "material_rohrle.yml",
    params:
        sim_dir=directory(PAPER_FIG_DIR / "{mesh}" / "growth_sim_long"),
        mesh_name="{mesh}",
        mesh_path=lambda wildcards: MESHES[wildcards.mesh],
        output_freq=POSTPROCESS_FREQ,
    conda:
        "musclex"
    resources:
        time="01:00:00",
        ntasks=8,
    shell:
        """
        export OMP_NUM_THREADS=1 && \
        mpirun -n {resources.ntasks} python {SCRIPT_DIR}/run_coupled_growth.py \
            --exercise-config {input.exercise_config} \
            --material-config {input.material_config} \
            --mesh-path {params.mesh_path} \
            --protocol-name long \
            --output-dir {params.sim_dir} \
            --output-freq {params.output_freq} \
        """


rule postprocessGrowth:
    localrule: True
    output:
        PAPER_FIG_DIR / "{mesh}" / "growth_sim_long" / "growth_rate.png",
    input:
        sim_dir=rules.runGrowthSimulation.output[0],
    conda:
        "musclex"
    shell:
        """
        python {SCRIPT_DIR}/plot_growth_diff.py {input.sim_dir} {output}
        """
