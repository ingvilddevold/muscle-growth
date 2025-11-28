# Supplementary code: Computational modeling of exercise-induced skeletal muscle hypertrophy through the IGF1-AKT signaling pathway

This repository contains the implementation of the coupled signaling-mechanics model of exercise-induced skeletal muscle growth in [...]

## Installation and dependencies
The project is based on [FEniCSx](https://fenicsproject.org) v0.10.0. 
The workflow management system [Snakemake](https://snakemake.readthedocs.io/en/stable/#) is used to run scripts. 

### Conda
Set up the Conda environment `musclex` with all required dependencies from the `environment.yml` file as
```bash
conda env create -f environment.yml
conda activate musclex
```

In addition, set up an environment with Snakemake (and the cluster plugin if running on e.g. ex3)
```bash
conda create --name snakemake_env -c conda-forge -c bioconda snakemake snakemake-executor-plugin-cluster-generic
```

### Docker
TO DO

## Use
There is one Snakefile in `scripts/` for each part of the paper results. 
* `figure3.smk`: ODE model only.
* `figure4.smk`: Coupled model with idealized geometry.
* `figure5.smk`: Coupled model with realistic geometries.
* `figure6.smk`: Coupled model with signaling heterogeneity.

In addition to the `musclex` environment, create an environment with Snakemake as 
```bash
conda create -n snakemake_env snakemake
```

With `snakemake_env` active, run the Snakefile either on ex3 as
```bash
snakemake -s scripts/figureX.smk --profile scripts/ex3
```
or, if running locally, 
```bash
snakemake -s scripts/figureX.smk --use-conda --cores 4
```
Snakemake will automatically activate the `musclex` environment for each rule.

## Demo scripts
In addition to the Snakefiles, there are a selection of demo scripts available in `demo/`. 
