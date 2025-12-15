# Supplementary code: Mechanochemical modeling of exercise-induced skeletal muscle hypertrophy
[![DOI](https://zenodo.org/badge/1105944660.svg)](https://doi.org/10.5281/zenodo.17902802)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the coupled signaling-mechanics model of exercise-induced skeletal muscle growth described in the paper 
>I.S. Devold, P. Rangamani and M.E. Rognes (2025) "Mechanochemical modeling of exercise-induced skeletal muscle hypertrophy".

The framework couples a tissue-level transversely isotropic hyperelastic model with a system of ODEs representing the IGF1-AKT-mTOR-FOXO signaling pathway. This multiscale approach links exercise-driven cellular signaling events to macroscopic volumetric growth.

**Key Features**
- **ODE signaling model**: Simulates protein synthesis/degradation via IGF1, AKT, mTOR, and FOXO dynamics
- **Finite element mechanical model**: Hyperelasticity with volumetric growth kinematics
- **Bidirectional coupling**: Signaling drives growth tensor; muscle cross-sectional area modulates protein synthesis rate
- **Geometries**: Supports both idealized and anatomically realistic meshes (from the Visible Human Dataset)



## Installation and Dependencies
The project is based on [FEniCSx](https://fenicsproject.org) v0.10.0. 
The workflow management system [Snakemake](https://snakemake.readthedocs.io/en/stable/#) is used to run scripts and reproduce the results. 

- **FEM & numerics**: `fenics-dolfinx`, `openmpi`, `scipy`, `numpy`
- **Mesh & I/O**: `gmsh`, `meshio`, `adios2`, `adios4dolfinx`
- **Plotting**: `matplotlib`, `seaborn`, `scienceplots`, `pyvista`
- **Utilities**: `pyyaml`, `typer`, `imageio`, `imageio-ffmpeg`
- **Sensitivity analysis**: `SALib`, `tqdm`
- **Workflow**: [Snakemake](https://snakemake.readthedocs.io/en/stable/#)

All dependencies are listed in `environment.yml`.


### Conda
Set up the main Conda environment `musclex` with all required dependencies:
```bash
conda env create -f environment.yml
conda activate musclex
```

For workflow management, set up a separate environment for Snakemake (and the cluster plugin if running on HPC systems like eX3):
```bash
conda create --name snakemake_env -c conda-forge -c bioconda snakemake snakemake-executor-plugin-cluster-generic
```


### Docker
The latest Docker image containing dependencies and the `musclex` package can be pulled from
```bash
docker pull ghcr.io/ingvilddevold/muscle-growth:main
```

## Usage

### Reproducing Paper Results

The workflow is organized into distinct Snakefiles in `scripts/`, each corresponding to a Results section and figure from the paper.

Before running, activate the `snakemake_env`:
```bash
conda activate snakemake_env
```

Run the specific Snakefile for the desired figure. Snakemake will automatically activate the `musclex` environment for computational rules.

| Snakefile | Description | Figure |
| :--- | :--- | :--- |
| `Snakefile_signaling.smk` | **Signaling Dynamics (ODE only)**<br>Simulates the time evolution of IGF1, AKT, FOXO, and mTOR under three exercise protocols (MWF, Weekly, Every-3-days). | Fig. 3 |
| `Snakefile_idealized.smk` | **Muscle Growth in Idealized Geometry**<br>Runs the full coupled model on an idealized fusiform geometry. Outputs include time-courses for Cross-Sectional Area (CSA), Volume, and the feedback-regulated protein synthesis rate $k_M$. | Fig. 4 |
| `Snakefile_realistic.smk` | **Realistic Geometries**<br>Simulates growth across 12 anatomically realistic meshes (Biceps Femoris, Semitendinosus, Tibialis Anterior) derived from the Visible Human Dataset. | Fig. 6 |
| `Snakefile_heterogeneity.smk` | **Signaling Heterogeneity**<br>Runs an ensemble simulation with spatially perturbed signaling parameters (IGF1, AKT, FOXO, mTOR) to assess the impact of local biological variability on macroscopic tissue deformation. Uses the female left BFLH geometry and the MWF protocol. | Fig. 7 |

#### Running the Snakefiles
To run on a cluster with slurm, use the provided profile:
```bash
snakemake -s scripts/Snakefile_XXX.smk --profile scripts/ex3
```
To run locally, use Conda and specify the number of cores:
```bash
snakemake -s scripts/Snakefile_XXX.smk --use-conda --cores 4
```
*Note: `Snakefile_realistic.smk` and `Snakefile_heterogeneity.smk` are intended for HPC (eX3) due to long runtimes. For local runs, especially for plotting, you may need to remove `xvfb-run` commands.*


### Demo Scripts
Standalone demo scripts in `demos/` can be run on a standard laptop:
- `demo_signaling_model.py`: Signaling model simulation for an example exercise protocol
- `demo_muscle_contraction.py`: Muscle contraction in an idealized fusiform geometry
- `demo_coupled_model.py`: Coupled signaling-mechanics model in an idealized fusiform geometry

These scripts are also available as tutorials in the [docs](https://ingvilddevold.github.io/muscle-growth/README.html).


## Realistic Muscle Geometries
The realistic muscle geometries used in this study were derived from the **Visible Human Dataset** as processed by Andreassen et al. (2023). 
A copy of the [original surface geometries](https://digitalcommons.du.edu/visiblehuman/) is included under `data/VHF_surfaces` (female) and `data/VHM_surfaces` (male), while the processed meshes and fiber fields required to run `Snakefile_realistic.smk` are available in the `meshes/` directory. 
These processed geometries include the **Biceps Femoris Long Head**, **Semitendinosus**, and **Tibialis Anterior** muscles, with four instances of each (male and female, left and right).

### Data License & Attribution
The geometry data included in `data/` are licensed under the Creative Commons Attribution 4.0 International License ([CC BY 4.0](http://creativecommons.org/licenses/by/4.0/)). Accordingly, please attribute the original datasets to the following works:

> Andreassen, T. E., et al. (2023). Three Dimensional Lower Extremity Musculoskeletal Geometry of the Visible Human Female and Male. *Scientific Data*, 10. [DOI: 10.1038/s41597-022-01905-2](https://doi.org/10.1038/s41597-022-01905-2)
> 
> Andreassen, T. E., et al. (2024).
"Automated 2D and 3D finite element overclosure adjustment and mesh morphing using generalized regression neural networks."
*Medical Engineering & Physics*, 126.
[DOI: 10.1016/j.medengphy.2024.104136](https://doi.org/10.1016/j.medengphy.2024.104136)

*Note: The software code in this repository remains licensed under the MIT License.* 

