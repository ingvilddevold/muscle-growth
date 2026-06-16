# Simulations and postprocessing for mesh convergence analysis
#
# Run script as:
#   $ snakemake -s scripts/Snakefile_convergence.smk --use-conda --cores 4

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

MESHES = {"idealizedRef0": ROOT_DIR / "meshes" / "muscle-idealized",
          "idealizedRef1": ROOT_DIR / "meshes" / "muscle-idealizedRef1",
          "idealizedRef2": ROOT_DIR / "meshes" / "muscle-idealizedRef2",
}

# --- Postprocessing Parameters ---
POSTPROCESS_FREQ = 1
POSTPROCESS_WARP_SCALE = 10.0


rule all:
    input:
        PAPER_FIG_DIR / "mesh_convergence.png",
        expand(
            PAPER_FIG_DIR / "{mesh}" / "growth_sim_{protocol}_postprocessed",
            mesh=MESHES.keys(),
            protocol=PROTOCOL,
        ),


rule runGrowthSimulation:
    output:
        sim_dir=directory(PAPER_FIG_DIR / "{mesh}" / "growth_sim_{protocol}"),
        results_csv=PAPER_FIG_DIR
        / "{mesh}"
        / "growth_sim_{protocol}"
        / "growth_results.csv",
    input:
        exercise_config=CONFIG_DIR / "exercise_eq_reduced_k1.yml",
        material_config=CONFIG_DIR / "material_rohrle.yml",
    params:
        mesh_name="{mesh}",
        mesh_path=lambda wildcards: MESHES[wildcards.mesh],
        protocol_name="{protocol}",
        output_freq=POSTPROCESS_FREQ,
    wildcard_constraints:
        protocol=PROTOCOL,
    conda:
        "musclex"
    resources:
        time="01:00:00",
    shell:
        """
        python {SCRIPT_DIR}/run_coupled_growth.py \
            --exercise-config {input.exercise_config} \
            --material-config {input.material_config} \
            --mesh-path {params.mesh_path} \
            --protocol-name {params.protocol_name} \
            --output-dir {output.sim_dir} \
            --output-freq {params.output_freq} \
        """


rule postprocessGrowth:
    output:
        directory(PAPER_FIG_DIR / "{mesh}" / "growth_sim_{protocol}_postprocessed"),
    input:
        sim_dir=rules.runGrowthSimulation.output[0],
        conf_file=rules.runGrowthSimulation.input.material_config,
    params:
        freq=POSTPROCESS_FREQ,
        warp_scale=POSTPROCESS_WARP_SCALE,
    wildcard_constraints:
        protocol=PROTOCOL,
    conda:
        "musclex"
    shell:
        "python {SRC_DIR}/postprocess_growth.py {input.sim_dir} {input.conf_file} --freq {params.freq} --warp-scale {params.warp_scale}"


rule plotMeshConvergence:
    input:
        # Pulls the results from all meshes
        csvs=expand(PAPER_FIG_DIR / "{mesh}" / "growth_sim_{protocol}" / "growth_results.csv", mesh=MESHES.keys(), protocol=PROTOCOL),
        script=SCRIPT_DIR / "plot_mesh_convergence.py",
    output:
        PAPER_FIG_DIR / "mesh_convergence.png",
    params:
        meshes=",".join(MESHES.keys()),
    conda:
        "musclex"
    shell:
        """
        python {input.script} \
            {input.csvs} \
            --mesh-names {params.meshes} \
            --output-file {output}
        """
    