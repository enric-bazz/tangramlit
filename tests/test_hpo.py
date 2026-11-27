"""Hyperparameter tuning: trials train with the same schedule as map_cells_to_space."""

import math

import pytest

import tangramlit as tgl

GENE_SPLIT = dict(
    train_genes_names=[f"gene{i}" for i in range(0, 6)],
    val_genes_names=[f"gene{i}" for i in range(6, 9)],
)
# Terms switched off (0 is never tuned) and fixed arguments of the toy study
FIXED = dict(
    filter=False, lambda_d=0, lambda_r=0, lambda_l1=0, lambda_l2=0, lambda_sparsity_g1=0,
    lambda_getis_ord=0, lambda_moran=0, lambda_geary=0, lambda_ct_islands=0,
)
SCHEDULE = dict(num_epochs=30, warmup_epochs=5, patience=5)


def test_trial_value_is_the_value_of_the_mapping_trained_with_its_params(tmp_path, monkeypatch, make_toy_pair):
    """A trial must be scored under the conditions its lambdas are later used in.

    Oracle: map_cells_to_space() run with the parameters of the trial and the same schedule
    reaches the value Optuna recorded for that trial.
    """
    monkeypatch.chdir(tmp_path)
    adata_sc, adata_st = make_toy_pair()
    best_value, best_params, trials = tgl.tune_loss_coefficients(
        adata_sc, adata_st, random_state=0, n_trials=1,
        lambda_g2=1, lambda_neighborhood_g1=1, **FIXED, **SCHEDULE, **GENE_SPLIT,
    )
    assert set(best_params) == {"lambda_g2", "lambda_neighborhood_g1"}

    adata_sc, adata_st = make_toy_pair()
    adata_map, _, _ = tgl.map_cells_to_space(
        adata_sc, adata_st, random_state=0, device="cpu", **best_params, **FIXED, **SCHEDULE, **GENE_SPLIT,
    )
    outcome = adata_map.uns["training_schedule"]

    assert outcome["monitor"] == "val_objective"
    assert best_value == pytest.approx(outcome["best_value"], rel=1e-5)
    assert best_value == pytest.approx(max(adata_map.uns["validation_history"]["val_objective"]), rel=1e-5)


def test_tuning_runs_without_a_random_state(tmp_path, monkeypatch, make_toy_pair):
    """random_state=None is the default and must not break the study."""
    monkeypatch.chdir(tmp_path)
    adata_sc, adata_st = make_toy_pair()
    best_value, best_params, trials = tgl.tune_loss_coefficients(
        adata_sc, adata_st, n_trials=2, lambda_g2=1, lambda_neighborhood_g1=0, **FIXED, **SCHEDULE, **GENE_SPLIT,
    )

    assert math.isfinite(best_value)
    assert len(trials) == 2
    assert set(best_params) == {"lambda_g2"}
