# Tangram Reimplementation in PyTorch Lightning

**Tangram** is a probabilistic mapping framework that aligns single-cell RNA-seq profiles and spatial transcriptomics measurements through a learned mapping matrix that assigns cells to spatial locations by optimizing a similarity-based objective between predicted and observed spatial gene expression.

This repository contains a reimplementation of Tangram in PyTorch Lightning that:

* Builds upon the original Tangram implementation by the Broad Institute: [https://github.com/broadinstitute/Tangram](https://github.com/broadinstitute/Tangram); and the refinement extension: [https://github.com/daisybio/Tangram_Refinement_Strategies](https://github.com/daisybio/Tangram_Refinement_Strategies)

* Provides a modular trainer and utilities for inspecting training dynamics
* Adds tools for downstream mapping analysis.

---

## Installation

```bash
pip install git+https://github.com/enric-bazz/tangramlit.git
```

To work on the package, clone the repository and run `uv sync`: it recreates the development
environment recorded in `uv.lock`. The tests run with `uv run pytest`, and `examples/` contains
scripts that walk through the public API.

---

## Supported Modes

Currently supported:

* `cell` mode as default
* `constrained` mode, controlled by the `filter` flag in the default mode

Not supported:

* `cluster` mode

---

## Intended Use Case

This implementation is designed for imaging-based spatial transcriptomics technologies that provide:

* Single-molecule resolution
* Single-cell spatial expression profiles

Accordingly:

* Only a uniform density prior over spots is implemented
* `rna_count_based` priors are not supported
* Custom density priors are not yet implemented

---

## Command-Line Interface (CLI)

The `tangramlit` package provides a CLI to run common workflows without writing Python code. Each command reads
two `.h5ad` files and a YAML configuration from an input directory and writes its results to an output directory:

```bash
# Train a spatial mapping model
tangramlit train -i /path/to/input_dir -sc sc_data.h5ad -ist spatial_data.h5ad -c config.yaml -o /path/to/output_dir --validate --test

# Tune loss hyperparameters
tangramlit tune -i /path/to/input_dir -sc sc_data.h5ad -ist spatial_data.h5ad -c config.yaml -o /path/to/output_dir --n-trials 100

# Cross-validate genes mapping
tangramlit cv -i /path/to/input_dir -sc sc_data.h5ad -ist spatial_data.h5ad -c config.yaml -o /path/to/output_dir --k-folds 5
```

The configuration file holds the keyword arguments of `map_cells_to_space` (loss coefficients, `filter`,
`num_epochs`, `random_state`, `cluster_label`, ...).

| Command | Output files |
|---|---|
| `train` | `adata_map.h5ad`, `training_results.yaml`; with `--validate` also `validation_results.yaml` and `spatial_gene_comparison.csv`; with `--test` also `test_results.yaml` |
| `tune` | `<study>_best_params.yaml`, `<study>_trials.csv`, and `<study>.db` unless `--storage` is given |
| `cv` | `cv_metrics.yaml`, `cv_metrics_per_fold.csv` |

Further details and options with:

```bash
tangramlit --help
tangramlit train --help
```

---

## Public API

In addition to the CLI, `tangramlit` exposes a Python API for interactive use in scripts or notebooks. It is largely consistent with the original Tangram API, with extensions for:

- Training, analysis, and inspection of mappings  
- Plotting and visualization  
- Mapping validation and benchmarking  
- Hyperparameter optimization.

A minimal mapping, with `adata_sc` and `adata_st` two AnnData objects (the spatial one with coordinates in `obsm['spatial']`):

```python
import tangramlit as tgl

# Training genes enter the loss, validation genes drive early stopping, test genes are only scored at the end
train_genes, val_genes, test_genes = tgl.split_train_val_test_genes(adata_sc, adata_st, random_state=0)
adata_map, mapper, datamodule = tgl.map_cells_to_space(
    adata_sc,
    adata_st,
    train_genes_names=train_genes,
    val_genes_names=val_genes,
    test_genes_names=test_genes,
    lambda_g1=1,
    lambda_d=1e-3,
    num_epochs=500,
    random_state=0,
)

# Score the mapping on the held-out test genes
test_results = tgl.test_mapping_experiment(mapper, datamodule)[0]

# Project single-cell expression onto space and score it gene by gene
adata_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
per_gene = tgl.compare_spatial_geneexp(adata_ge, datamodule)       # cosine similarity score
per_gene = tgl.benchmark_mapping(adata_ge, datamodule, per_gene)   # adds PCC, RMSE, JS, SSIM
```

Notes:

- Mapping modifies the input AnnData objects in place: gene names are converted to lower case,
  cells or spots with missing labels or coordinates are removed, and preprocessing annotations are added.
- Plotting functions return their matplotlib figure(s) and do not call `plt.show()`: figures are displayed
  automatically in notebooks, while scripts have to save or show them.
