"""
Example script for the Optuna tuning public API.
Run it from anywhere: python examples/optuna_tuning.py
"""

import sys
import logging
import yaml
import optuna
import anndata as ad
import tangramlit as tgl
from pathlib import Path

# Example data (not tracked by git): put the files in <repository>/data
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OUT_DIR = Path(__file__).resolve().parents[1] / "out"


def main():
    # Set study parameters
    adata_sc = ad.read_h5ad(DATA_DIR / "test_sc.h5ad")
    adata_st = ad.read_h5ad(DATA_DIR / "test_slice200.h5ad")
    study_name = "api_test_study"
    OUT_DIR.mkdir(exist_ok=True)
    storage = f"sqlite:///{(OUT_DIR / 'tangram_tuning_api_test.db').as_posix()}"
    n_trials = 2    
    resume = False

    with open(DATA_DIR / "train_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    random_state = config["random_state"]

    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

    # Get train/val genes (0.8 split)
    train_genes, val_genes, _ = tgl.split_train_val_test_genes(adata_sc, adata_st, random_state=random_state)

    best_value, best_params, trials_df = tgl.tune_loss_coefficients(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=None,
        train_genes_names=train_genes,
        val_genes_names=val_genes,
        **config,
        study_name=study_name,
        storage=storage,
        n_trials=n_trials,
        resume=True,
        )

    with open(OUT_DIR / f"{study_name}_best_params.yaml", "w") as f:
            yaml.dump(best_params, f, default_flow_style=False)

    trials_df.to_csv(OUT_DIR / f"{study_name}_trials.csv", index=False, sep=';')
 

if __name__ == "__main__":
    main()