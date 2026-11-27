"""Held-out objective: the unweighted mean of the active similarity terms, scored on held-out genes."""

import numpy as np
import pytest

import tangramlit as tgl

GENE_SPLIT = dict(
    train_genes_names=[f"gene{i}" for i in range(0, 6)],
    val_genes_names=[f"gene{i}" for i in range(6, 9)],
    test_genes_names=[f"gene{i}" for i in range(9, 12)],
)


def _mean_cosine(a, b, axis):
    return np.mean((a * b).sum(axis) / (np.linalg.norm(a, axis=axis) * np.linalg.norm(b, axis=axis)))


def test_objective_is_the_score_when_only_the_main_term_is_active(tmp_path, monkeypatch, make_toy_pair):
    """With lambda_g1 alone the objective must reduce to the plain score, on validation and on test genes."""
    adata_sc, adata_st = make_toy_pair()
    monkeypatch.chdir(tmp_path)
    _, model, datamodule = tgl.map_cells_to_space(
        adata_sc, adata_st, num_epochs=3, random_state=0, device="cpu", **GENE_SPLIT
    )

    val = tgl.validate_mapping_experiment(model, datamodule)[0]
    test = tgl.test_mapping_experiment(model, datamodule)[0]

    assert val["val_objective"] == pytest.approx(val["val_score"], rel=1e-6)
    assert test["test_objective"] == pytest.approx(test["test_score"], rel=1e-6)


def test_objective_averages_the_active_terms_without_their_lambdas(tmp_path, monkeypatch, make_toy_pair):
    """The objective must not depend on the lambda values, only on which terms are active.

    Oracle: the gene-voxel, voxel-gene, neighborhood and Moran similarities recomputed with NumPy
    from the saved mapping and the spatial graph, on the test genes, then averaged with equal weights.
    The lambdas are deliberately far from 1: a lambda-weighted objective would not match.
    """
    adata_sc, adata_st = make_toy_pair()
    monkeypatch.chdir(tmp_path)
    adata_map, model, datamodule = tgl.map_cells_to_space(
        adata_sc, adata_st, num_epochs=3, random_state=0, device="cpu",
        lambda_g1=1, lambda_g2=0.3, lambda_neighborhood_g1=0.05, lambda_moran=7, **GENE_SPLIT
    )
    test = tgl.test_mapping_experiment(model, datamodule)[0]

    test_genes = GENE_SPLIT["test_genes_names"]
    S = datamodule.adata_sc[:, test_genes].X
    G = datamodule.adata_st[:, test_genes].X
    G_pred = adata_map.X.T @ S

    conn = datamodule.adata_st.obsp["spatial_connectivities"].toarray()
    dist = datamodule.adata_st.obsp["spatial_distances"].toarray()
    W = conn * (dist / dist.sum(axis=1, keepdims=True))  # row-standardized distances on the neighbor links

    def moran(X):
        z = X - X.mean(axis=0)
        return X.shape[0] * z * (W @ z) / (z * z).sum(axis=0)

    W_self = W + np.eye(W.shape[0])
    expected = np.mean([
        _mean_cosine(G_pred, G, axis=0),                    # gene-voxel
        _mean_cosine(G_pred, G, axis=1),                    # voxel-gene
        _mean_cosine(W_self @ G_pred, W_self @ G, axis=0),  # neighborhood
        _mean_cosine(moran(G_pred), moran(G), axis=0),      # Moran's I
    ])

    assert test["test_objective"] == pytest.approx(expected, rel=1e-4)
    assert test["test_objective"] != pytest.approx(test["test_score"], rel=1e-3)
