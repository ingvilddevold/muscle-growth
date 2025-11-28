# Simulations and postprocessing for Figure 4
#   contraction and growth simulations in idealized fusiform geometry
#
# Run script as e.g.:
#   $ snakemake -s scripts/figure4.smk --cores 4 --use-conda

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
PROTOCOLS = ["defreitas", "everythreedays", "weekly"]
MESHES = {"idealized_fusiform": ROOT_DIR / "meshes" / "muscle-idealized"}

# --- Postprocessing Parameters ---
POSTPROCESS_FREQ = 10
POSTPROCESS_WARP_SCALE = 10.0


rule all:
    input:
        expand(PAPER_FIG_DIR / "{mesh}" / "growth_comparison.png", mesh=MESHES.keys()),
        expand(PAPER_FIG_DIR / "{mesh}" / "mechanics_postprocessed", mesh=MESHES.keys()),
        expand(
            PAPER_FIG_DIR / "{mesh}" / "growth_sim_{protocol}_postprocessed",
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
        ),
        expand(
            PAPER_FIG_DIR / "{mesh}" / "growth_comparison_3d.png", mesh=MESHES.keys()
        ),


# --- Simulation Rule 1: Run active contraction ---
rule runMechanicsSimulation:
    output:
        directory(PAPER_FIG_DIR / "{mesh}" / "mechanics"),
    input:
        config=CONFIG_DIR / "material_rohrle.yml",
    conda:
        "musclex"
    params:
        mesh_name="{mesh}",
        mesh_path=lambda wildcards: MESHES[wildcards.mesh],
    shell:
        """
        python {SCRIPT_DIR}/run_contraction.py \
            --output-dir {output} \
            --material-config {input.config} \
            --mesh-name {params.mesh_name} \
            --mesh-path {params.mesh_path} \
            --no-is-realistic
        """


rule postprocessMechanics:
    output:
        directory(PAPER_FIG_DIR / "{mesh}" / "mechanics_postprocessed"),
    input:
        sim_dir=rules.runMechanicsSimulation.output[0],
        conf_file=rules.runMechanicsSimulation.input.config,
        script=SRC_DIR / "postprocess_mechanics.py",
    conda:
        "musclex"
    shell:
        "python {input.script} {input.sim_dir} {input.conf_file}"


# --- Simulation Rule 2: Run full growth model ---
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
        protocol="|".join(PROTOCOLS),
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
        protocol="|".join(PROTOCOLS),
    conda:
        "musclex"
    shell:
        "python {SRC_DIR}/postprocess_growth.py {input.sim_dir} {input.conf_file} --freq {params.freq} --warp-scale {params.warp_scale}"


rule plotGrowthComparison:
    input:
        # This rule needs all the growth CSVs
        expand(
            rules.runGrowthSimulation.output.results_csv,
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
        ),
        script= SCRIPT_DIR / "plot_growth_comparison.py",
    output:
        PAPER_FIG_DIR / "{mesh}" / "growth_comparison.png",
    params:
        protocols=",".join(PROTOCOLS),
    conda:
        "musclex"
    shell:
        """
        python {input.script} \
            {input} \
            --protocols {params.protocols} \
            --output-file {output}
        """


rule plot3dGrowthComparison:
    input:
        mesh_file=lambda wildcards: MESHES["idealized_fusiform"],
        sim_dirs=expand(
            rules.runGrowthSimulation.output.sim_dir,
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
        ),
    output:
        PAPER_FIG_DIR / "{mesh}" / "growth_comparison_3d.png",
    params:
        protocols=",".join(PROTOCOLS),
        warp_scale=POSTPROCESS_WARP_SCALE,
        sim_dirs_formatted=lambda wildcards, input: " ".join(
            [f"--sim-dirs {d}" for d in input.sim_dirs]
        ),
    conda:
        "musclex"
    shell:
        """
        python {SCRIPT_DIR}/plot_3d_growth_comparison.py \
            --mesh-file {input.mesh_file} \
            {params.sim_dirs_formatted} \
            --protocols '{params.protocols}' \
            --output-file {output} \
            --warp-scale {params.warp_scale} \
        """
