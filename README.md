# CtxBandit

This repository provides the source code for the paper **[Bayesian Inference of Contextual Bandit Policies via Empirical Likelihood](http://jmlr.org/papers/v27/23-0958.html)**, published in *JMLR*.

---

## Repository Structure

```
examples/        – Notebooks demonstrating how to build and run inference models
experiments/     – Code for reproducing experiments in the paper  
src/ctxbandit/   – Source code of the ctxbandit package
pyproject.toml   – Package configuration and dependency specifications
```

---

## Installation

Create a new environment and install the package locally:

```bash
mamba create -n ctxbandit python=3.12
mamba activate ctxbandit

git clone https://github.com/jr-ouyang/ctxbandit.git
cd ctxbandit
pip install -e .
```
All dependencies are installed automatically as specified in [pyproject.toml](pyproject.toml).

---

## Quick Start

To get started, see the Jupyter notebooks in the [`examples/`](examples) directory.

These notebooks demonstrate how to construct inference models for a single policy, joint policies, and policy value differences.

---

## Experiments

The [`experiments/`](experiments) directory contains experiments reported in the paper.  

The workflow is managed using [Snakemake 9.13.3](https://snakemake.github.io/) together with the [Snakemake executor plugin for Slurm 1.8.0](https://github.com/snakemake/snakemake-executor-plugin-slurm). 

The rules and configurations for Slurm job submission are defined in the [Snakefile](experiments/Snakefile).

