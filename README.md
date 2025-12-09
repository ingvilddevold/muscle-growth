# Supplementary code: Computational modeling of exercise-induced skeletal muscle hypertrophy through the IGF1-AKT signaling pathway

This repository contains the implementation of the coupled signaling-mechanics model of exercise-induced skeletal muscle growth described in the paper "Computational modeling of exercise-induced skeletal muscle hypertrophy through the IGF1-AKT signaling pathway".

The framework couples a tissue-level transversely isotropic hyperelastic model with a system of ordinary differential equations (ODEs) representing the IGF1-AKT-mTOR-FOXO signaling pathway. This multiscale approach links cellular signaling event, driven by exercise protocols, to macroscopic volumetric growth.

**Features**
* Signaling Model: ODE system simulating protein synthesis/degradation balance via IGF1, AKT, mTOR, and FOXO dynamics.
* Mechanical Model: Finite element implementation of hyperelasticity with volumetric growth kinematics.
* Coupling: Bidirectional feedback where signaling drives the growth tensor and muscle cross-sectional area (CSA) modulates protein synthesis rates.
* Geometries: Support for both idealized fusiform and anatomically realistic meshes (derived from the Visible Human Dataset).


## Installation and dependencies
The project is based on [FEniCSx](https://fenicsproject.org) v0.10.0. 
The workflow management system [Snakemake](https://snakemake.readthedocs.io/en/stable/#) is used to run scripts and reproduce the results. 

### Conda
Set up the main Conda environment `musclex` with all required dependencies (FEniCSx, SciPy, etc.) from the `environment.yml` file:
```bash
conda env create -f environment.yml
conda activate musclex
```

In addition, set up a separate environment for Snakemake (including the cluster plugin if running on HPC systems like eX3):
```bash
conda create --name snakemake_env -c conda-forge -c bioconda snakemake snakemake-executor-plugin-cluster-generic
```

### Docker
Coming soon


## Usage
The reproduction workflow is organized into distinct Snakefiles located in scripts/, corresponding to the key results presented in the paper.

In addition to the musclex environment (which handles the physics/math), ensure you have the `snakemake_env` active:
```bash
conda activate snakemake_env
```

### Reproducing results
Run the specific Snakefile for the desired figure. Snakemake will automatically handle the activation of the `musclex` environment for the computational rules. 

| Snakefile | Description | Corresponding Figure |
| :--- | :--- | :--- |
| `Snakefile_signaling.smk` | **Signaling Dynamics (ODE only)**<br>Simulates the time evolution of IGF1, AKT, FOXO, and mTOR under three exercise protocols (MWF, Weekly, Every-3-days). | Figure 3 |
| `Snakefile_idealized.smk` | **Muscle Growth in Idealized Geometry**<br>Runs the full coupled model on an idealized fusiform geometry. Outputs include time-courses for Cross-Sectional Area (CSA), Volume, and the feedback-regulated protein synthesis rate $k_M$. | Figure 4 |
| `Snakefile_realistic.smk` | **Realistic Geometries**<br>Simulates growth across 12 anatomically realistic meshes (Biceps Femoris, Semitendinosus, Tibialis Anterior) derived from the Visible Human Dataset. | Figure 6 |
| `Snakefile_heterogeneity.smk` | **Signaling Heterogeneity**<br>Runs an ensemble simulation with spatially perturbed signaling parameters (IGF1, AKT, FOXO, mTOR) to assess the impact of local biological variability on macroscopic tissue deformation. Uses the female left BFLH geometry and the MWF protocol. | Figure 7 |

#### Running the Snakefiles
To run on a cluster using the slurm executor (or similar), use the provided profile:
```bash
snakemake -s scripts/Snakefile_XXX.smk --profile scripts/ex3
```
If running locally, specify to use Conda and set the number of cores
```bash
snakemake -s scripts/Snakefile_XXX.smk --use-conda --cores 4
```
Note that `Snakefile_realistic.smk` and `Snakefile_heterogeneity.smk` intend to be run on eX3 due to long runtimes. To run those locally, in particular the plot scripts, you may need to remove the `xvfb-run` commands.

## Demo scripts
In addition to the Snakefiles, there are a selection of standalone demo scripts available in `demos/`, all feasible for testing on a standard laptop:
* `demo_signaling.py` – Signaling model simulation for an example exercise protocol.
* `demo_muscle_contraction.py` – Simulating muscle contraction in an idealized fusiform muscle geometry.
* `demo_coupled.py` – Simulating the coupled signaling-mechanics model in an idealized fusiform muscle geometry.


## Geometries
The realistic muscle geometries used in this study were derived from the [Visible Human Dataset](https://digitalcommons.du.edu/visiblehuman/). The processed meshes and fiber fields required to run `Snakefile_realistic.smk` are available in the `meshes/` directory. These include the Biceps Femoris Long Head, Semitendinosus and Tibialis Anterior leg muscles, with four instances of each (male and female, left and right).
