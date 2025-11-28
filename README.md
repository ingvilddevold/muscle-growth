# Muscle growth

This repository contains code associated with the preprint ...


## Installation and dependencies
The project relies on FEniCSx v0.10.0. 

### Conda
Set up a Conda environment, named `musclex`, with all dependencies from the `environment.yml` file as
```bash
conda env create -f environment.yml
conda activate musclex
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
or if running locally, 
```bash
snakemake -s scripts/figureX.smk --use-conda
```

## Demo scripts
In addition to the Snakefiles, there are a selection of demo scripts available in `demo/`. 