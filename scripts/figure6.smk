#
# Simulations and postprocessing for Figure 6
#   coupled growth simulations with spatially varying ODEs
#
# Run as
#   $ snakemake -s scripts/figure6.smk --profile scripts/ex3
#

from pathlib import Path
import json

# --- Global Configuration ---
ROOT_DIR = Path(workflow.basedir).parent  # Path to repo
print(f"Root dir: {ROOT_DIR}")
RESULTS_DIR = ROOT_DIR / "results"
CONFIG_DIR = ROOT_DIR / "config_files"
SCRIPT_DIR = ROOT_DIR / "scripts"
SRC_DIR = ROOT_DIR / "src" / "musclex"
# Path for outputs
PAPER_FIG_DIR = RESULTS_DIR / "figure6"
PAPER_FIG_DIR.mkdir(exist_ok=True, parents=True)


# --- Define Simulations ---
# Protocols to simulate
# PROTOCOLS = ["testing"]
PROTOCOLS = ["defreitas"]

# Meshes to simulate
MESH_DIR = ROOT_DIR / "meshes"
MESHES = {
    "BFLH_VHF_Left": MESH_DIR / "VHF_Left_Muscle_BicepsFemorisLong_smooth",
}

# Variations (seeds)
VARIATIONS = {
    "baseline": 0,  # no variation
    "var1": 1,
    "var2": 2,
    "var3": 3,
    "var4": 4,
    "var5": 5,
}
variation_magnitude = 0.01  # 1% variation

# --- Postprocessing Parameters ---
POSTPROCESS_FREQ = 10
POSTPROCESS_WARP_SCALE = 10.0


# --- Target Rule ---
rule all:
    input:
        expand(
            PAPER_FIG_DIR
            / "{mesh}"
            / "growth_sim_{protocol}_{variation}_postprocessed",
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
            variation=VARIATIONS.keys(),
        ),
        expand(
            PAPER_FIG_DIR / "{mesh}" / "{protocol}_variation_comparison.png",
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
        ),
        expand(
            PAPER_FIG_DIR / "{mesh}" / "{protocol}_ODE_states.png",
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
        ),
        expand(
            PAPER_FIG_DIR / "{mesh}" / "{protocol}_{variation}_param_distribution.png",
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
            variation=VARIATIONS.keys(),
        ),
        expand(
            PAPER_FIG_DIR / "{mesh}" / "{protocol}_3d_comparison_grid.png",
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
        ),


# --- Simulation Rule: Run spatially-varying ODE model ---
rule runSpatialGrowthSimulation:
    output:
        sim_dir=directory(
            PAPER_FIG_DIR / "{mesh}" / "growth_sim_{protocol}_{variation}"
        ),
        results_csv=PAPER_FIG_DIR
        / "{mesh}"
        / "growth_sim_{protocol}_{variation}"
        / "growth_results.csv",
        ode_history_bp=directory(PAPER_FIG_DIR
        / "{mesh}"
        / "growth_sim_{protocol}_{variation}"
        / "ode_spatial_history.bp"),
        spatial_params_csv=PAPER_FIG_DIR
        / "{mesh}"
        / "growth_sim_{protocol}_{variation}"
        / "spatial_parameters.csv",
    input:
        script=SCRIPT_DIR / "run_coupled_growth.py",
        exercise_config=CONFIG_DIR / "exercise_eq_reduced_k1.yml",
        material_config=CONFIG_DIR / "material_rohrle.yml",
    params:
        mesh_name="{mesh}",
        mesh_path=lambda wildcards: MESHES[wildcards.mesh],
        protocol_name="{protocol}",
        output_freq=POSTPROCESS_FREQ,
        realistic_flag=lambda wildcards: (
            "--is-realistic" if "VH" in wildcards.mesh else ""
        ),
        spatial_flag="--is-spatial",
        variation_mag=lambda wildcards: (
            variation_magnitude if wildcards.variation != "baseline" else 0.0
        ),
        seed=lambda wildcards: VARIATIONS[wildcards.variation],
    wildcard_constraints:
        protocol="|".join(PROTOCOLS),
        variation="|".join(VARIATIONS.keys()),
    resources:
        time="06:00:00",
    conda:
        "musclex"
    shell:
        """
        python {input.script} \
            --exercise-config {input.exercise_config} \
            --material-config {input.material_config} \
            --mesh-path {params.mesh_path} \
            --protocol-name {params.protocol_name} \
            --output-dir {output.sim_dir} \
            --output-freq {params.output_freq} \
            {params.realistic_flag} \
            --is-spatial \
            --variation-magnitude {params.variation_mag} \
            --seed {params.seed}
        """


rule postprocessGrowth:
    output:
        directory(
            PAPER_FIG_DIR
            / "{mesh}"
            / "growth_sim_{protocol}_{variation}_postprocessed"
        ),
    input:
        sim_dir=rules.runSpatialGrowthSimulation.output[0],
        conf_file=rules.runSpatialGrowthSimulation.input.material_config,
    params:
        freq=POSTPROCESS_FREQ,
        warp_scale=POSTPROCESS_WARP_SCALE,
    wildcard_constraints:
        protocol="|".join(PROTOCOLS),
    conda:
        "musclex"
    shell:
        "xvfb-run -a python {SRC_DIR}/postprocess_growth.py {input.sim_dir} {input.conf_file} --freq {params.freq} --warp-scale {params.warp_scale}"


rule plotVariationComparison:
    output:
        plot=PAPER_FIG_DIR / "{mesh}" / "{protocol}_variation_comparison.png",
    input:
        # --- Collects all CSVs for a given mesh/protocol ---
        csvs=expand(
            rules.runSpatialGrowthSimulation.output.results_csv,
            mesh="{mesh}",
            protocol="{protocol}",
            variation=VARIATIONS.keys(),
        ),
        script=SCRIPT_DIR / "plot_variation_comparison.py",
    params:
        # Pass the variation names and paths to the script
        variation_labels=",".join(VARIATIONS.keys()),
        input_files=lambda wildcards, input: ",".join(input.csvs),
    conda:
        "musclex"
    resources:
        time="00:30:00",
    shell:
        """
        python {input.script} \
            --input-csvs {params.input_files} \
            --variation-labels {params.variation_labels} \
            --output-file {output.plot} \
            --protocol {wildcards.protocol}
        """


rule plotParameterDistribution:
    localrule: True
    output:
        plot=PAPER_FIG_DIR / "{mesh}" / "{protocol}_{variation}_param_distribution.png",
    input:
        csv=PAPER_FIG_DIR
        / "{mesh}"
        / "growth_sim_{protocol}_{variation}"
        / "spatial_parameters.csv",
        config_file=CONFIG_DIR / "exercise_eq_reduced_k1.yml",
        script=SCRIPT_DIR / "plot_param_distribution.py",
    wildcard_constraints:
        protocol="|".join(PROTOCOLS),
        variation="|".join(VARIATIONS.keys()),
    conda:
        "musclex"
    shell:
        """
        python {input.script} \
            --param-csv {input.csv} \
            --config-file {input.config_file} \
            --output-file {output.plot}
        """


rule plot3DGrid:
    localrule: True
    output:
        plot=PAPER_FIG_DIR / "{mesh}" / "{protocol}_3d_comparison_grid.png",
    input:
        # Collect sim_dirs for baseline and all variations
        sim_dirs=expand(
            rules.runSpatialGrowthSimulation.output.sim_dir,
            mesh="{mesh}",
            protocol="{protocol}",
            variation=VARIATIONS.keys(),
        ),
        script=SCRIPT_DIR / "plot_3d_heterogeneous.py",
    params:
        # Pass labels in specific order (baseline first)
        labels=",".join(
            ["baseline"] + sorted([v for v in VARIATIONS if v != "baseline"])
        ),
        input_dirs=lambda wildcards, input: ",".join(input.sim_dirs),
        ode_bp_filename="ode_spatial_history.bp", # Relative name within sim_dir
    wildcard_constraints:
        protocol="|".join(PROTOCOLS),
    conda:
        "musclex"
    resources:
        time="00:20:00",
    shell:
        """
        xvfb-run -a python {input.script} \
            --sim-dirs {params.input_dirs} \
            --labels {params.labels} \
            --output-file {output.plot} \
            --ode-bp-filename {params.ode_bp_filename} \
            --warp-scale {POSTPROCESS_WARP_SCALE} \
            --save-individual
        """

rule aggregateODEStates:
    output:
        csv=PAPER_FIG_DIR / "{mesh}" / "{protocol}_ode_states_aggregated.csv",
    input:
        # Collects all bps for a given mesh/protocol
        ode_bps=expand(
            rules.runSpatialGrowthSimulation.output.ode_history_bp,
            mesh="{mesh}",
            protocol="{protocol}",
            variation=VARIATIONS.keys(),
        ),
        script=SCRIPT_DIR / "plot_spatial_ode_states.py",
    params:
        # Pass the variation names and paths to the script
        variation_labels=",".join(VARIATIONS.keys()),
        ode_bp_files=lambda wildcards, input: ",".join(input.ode_bps),
    wildcard_constraints:
        protocol="|".join(PROTOCOLS),
    conda:
        "musclex"
    resources:
        time="03:00:00",
    shell:
        """
        python {input.script} aggregate \
            --ode-bp-files {params.ode_bp_files} \
            --variation-labels {params.variation_labels} \
            --output-csv {output.csv}
        """

rule plotODEStates:
    localrule: True
    output:
        plot=PAPER_FIG_DIR / "{mesh}" / "{protocol}_ODE_states.png",
        svg=PAPER_FIG_DIR / "{mesh}" / "{protocol}_ODE_states.svg",
    input:
        # Depends on the CSV from the aggregation step
        csv=rules.aggregateODEStates.output.csv,
        script=SCRIPT_DIR / "plot_spatial_ode_states.py",
    wildcard_constraints:
        protocol="|".join(PROTOCOLS),
    conda:
        "musclex"
    resources:
        time="00:10:00",
    shell:
        """
        python {input.script} plot \
            --input-csv {input.csv} \
            --output-file {output.plot}
        """
