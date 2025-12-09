# Simulations and postprocessing for Figure 5
#   contraction and growth simulations on realistic meshes
#
# Run the Snakefile as:
#   $ snakemake -s scripts/figure5.smk --profile scripts/ex3


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
PAPER_FIG_DIR = RESULTS_DIR / "figure5"
PAPER_FIG_DIR.mkdir(exist_ok=True, parents=True)


# --- Define Simulations ---
# Exercise protocols to simulate
PROTOCOLS = ["defreitas", "everythreedays", "weekly"]

# Meshes to simulate
MESH_DIR = ROOT_DIR / "meshes"
MESHES = {
    "BFLH_VHF_Right": MESH_DIR / "VHF_Right_Muscle_BicepsFemorisLongHead_smooth",
    "BFLH_VHF_Left": MESH_DIR / "VHF_Left_Muscle_BicepsFemorisLong_smooth",
    "BFLH_VHM_Right": MESH_DIR / "VHM_Right_Muscle_BicepsFemorisLong_smooth",
    "BFLH_VHM_Left": MESH_DIR / "VHM_Left_Muscle_BicepsFemorisLongus_smooth",
    "ST_VHF_Right": MESH_DIR / "VHF_Right_Muscle_Semitendonosus_smooth",
    "ST_VHF_Left": MESH_DIR / "VHF_Left_Muscle_Semitendinosus_smooth",
    "ST_VHM_Right": MESH_DIR / "VHM_Right_Muscle_Semitendinosus_smooth",
    "ST_VHM_Left": MESH_DIR / "VHM_Left_Muscle_Semitendonosus_smooth",
    "TA_VHF_Right": MESH_DIR / "VHF_Right_Muscle_TibialisAnterior_smooth",
    "TA_VHF_Left": MESH_DIR / "VHF_Left_Muscle_TibialisAnterior_smooth",
    "TA_VHM_Right": MESH_DIR / "VHM_Right_Muscle_TibialisAnterior_smooth",
    "TA_VHM_Left": MESH_DIR / "VHM_Left_Muscle_TibialisAnterior_smooth",
}

# --- Postprocessing Parameters ---
POSTPROCESS_FREQ = 10
POSTPROCESS_WARP_SCALE = 10.0

# Mapping from mesh names to muscle groups for postprocessing
MUSCLE_GROUPS = {
    "BFLH_VHF_Right": "BFLH",
    "BFLH_VHF_Left": "BFLH",
    "BFLH_VHM_Right": "BFLH",
    "BFLH_VHM_Left": "BFLH",
    "ST_VHF_Right": "ST",
    "ST_VHF_Left": "ST",
    "ST_VHM_Right": "ST",
    "ST_VHM_Left": "ST",
    "TA_VHF_Right": "TA",
    "TA_VHF_Left": "TA",
    "TA_VHM_Right": "TA",
    "TA_VHM_Left": "TA",
}


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
        PAPER_FIG_DIR / "CSA_comparison.png",
        PAPER_FIG_DIR / "Volume_comparison.png",
        PAPER_FIG_DIR / "mesh_statistics.csv",
        PAPER_FIG_DIR / "mesh_statistics.png",


# --- Simulation Rule 1: Run active contraction ---
rule runMechanicsSimulation:
    output:
        directory(PAPER_FIG_DIR / "{mesh}" / "mechanics"),
    input:
        config=CONFIG_DIR / "material_rohrle.yml",
    params:
        mesh_name="{mesh}",
        mesh_path=lambda wildcards: MESHES[wildcards.mesh],
        realistic_flag=lambda wildcards: (
            "--is-realistic"
            if "idealized" not in wildcards.mesh
            else "--no-is-realistic"
        ),
    conda:
        "musclex"
    resources:
        time="01:00:00",
    shell:
        """
        python -u {SCRIPT_DIR}/run_contraction.py \
            --output-dir {output} \
            --material-config {input.config} \
            --mesh-name {params.mesh_name} \
            --mesh-path {params.mesh_path} \
            {params.realistic_flag}
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
        "xvfb-run -a python {SRC_DIR}/postprocess_mechanics.py {input.sim_dir} {input.conf_file}"


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
        is_realistic=lambda wildcards: False if "idealized" in wildcards.mesh else True,
    wildcard_constraints:
        protocol="|".join(PROTOCOLS),
    conda:
        "musclex"
    resources:
        time="08:00:00",
    shell:
        """
        python {SCRIPT_DIR}/run_coupled_growth.py \
            --exercise-config {input.exercise_config} \
            --material-config {input.material_config} \
            --mesh-path {params.mesh_path} \
            --protocol-name {params.protocol_name} \
            --output-dir {output.sim_dir} \
            --output-freq {params.output_freq} \
            --is-realistic
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
        "xvfb-run -a python {SRC_DIR}/postprocess_growth.py {input.sim_dir} {input.conf_file} --freq {params.freq} --warp-scale {params.warp_scale}"


rule plotGrowthComparison:
    input:
        # This now uses the {mesh} wildcard to only gather results
        # for the specific mesh being processed by this job.
        expand(
            rules.runGrowthSimulation.output.results_csv,
            mesh="{mesh}",
            protocol=PROTOCOLS,
        ),
        script=f"{SCRIPT_DIR}/plot_growth_comparison.py",
    output:
        PAPER_FIG_DIR / "{mesh}" / "growth_comparison.png",
    params:
        protocols=",".join(PROTOCOLS),
    conda:
        "musclex"
    resources:
        time="00:05:00",
    shell:
        """
        python {input.script} \
            {input} \
            --protocols {params.protocols} \
            --output-file {output}
        """


rule plot3dGrowthComparison:
    input:
        mesh_file=lambda wildcards: MESHES[wildcards.mesh],
        sim_dirs=expand(
            rules.runGrowthSimulation.output.sim_dir,
            mesh="{mesh}",
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
    resources:
        time="00:05:00",
    shell:
        """
        xvfb-run -a python {SCRIPT_DIR}/plot_3d_growth_comparison.py \
            --mesh-file {input.mesh_file} \
            {params.sim_dirs_formatted} \
            --protocols '{params.protocols}' \
            --output-file {output} \
            --warp-scale {params.warp_scale} \
        """


rule plotQuantityComparison:
    localrule: True
    input:
        results_csv=expand(
            rules.runGrowthSimulation.output.results_csv,
            mesh=MESHES.keys(),
            protocol=PROTOCOLS,
        ),
        script=f"{SCRIPT_DIR}/plot_quantity_comparison.py",
    output:
        PAPER_FIG_DIR / "{quantity}_comparison.png",
    params:
        # Convert complex Python objects to strings for the command line
        muscle_groups=json.dumps(MUSCLE_GROUPS),
        protocols=",".join(PROTOCOLS),
        geometries=",".join(list(MESHES.keys())),
        quantity_to_plot="{quantity}",
    wildcard_constraints:
        quantity="CSA|Volume",
    conda:
        "musclex"
    resources:
        time="00:05:00",
    shell:
        """
        python {input.script} {input.results_csv} \
            --output-file {output} \
            --muscle-groups-json '{params.muscle_groups}' \
            --protocols-str "{params.protocols}" \
            --geometries-str "{params.geometries}" \
            --quantity-to-plot {params.quantity_to_plot}
        """


rule meshStatistics:
    localrule: True
    input:
        mesh_files=expand(MESHES[m] for m in MESHES.keys()),
    output:
        output_file=PAPER_FIG_DIR / "mesh_statistics.csv",
    params:
        mesh_list=lambda wildcards, input: ",".join(input.mesh_files),
    conda:
        "musclex"
    resources:
        time="00:10:00",
    shell:
        """
        python {SCRIPT_DIR}/mesh_statistics.py \
            --mesh-files '{params.mesh_list}' \
            --output-file {output.output_file}
        """


rule plotMeshStatistics:
    localrule: True
    input:
        csv_file=rules.meshStatistics.output.output_file,
        script=f"{SCRIPT_DIR}/plot_mesh_statistics.py",
    output:
        plot_file=PAPER_FIG_DIR / "mesh_statistics.png",
    conda:
        "musclex"
    resources:
        time="00:05:00",
    shell:
        """
        python {SCRIPT_DIR}/plot_mesh_statistics.py \
            --input-file {input.csv_file} \
            --output-file {output.plot_file} 
        """
