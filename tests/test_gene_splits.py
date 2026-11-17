"""Train / validation / test gene splits: the test genes are held out and scored by test_step."""

import numpy as np
import pytest

import tangramlit as tgl

GENE_SPLIT = dict(
    train_genes_names=[f"gene{i}" for i in range(0, 6)],
    val_genes_names=[f"gene{i}" for i in range(6, 9)],
    test_genes_names=[f"gene{i}" for i in range(9, 12)],
)


def test_test_step_scores_the_test_genes(tmp_path, monkeypatch, make_toy_pair):
    """test_step must report on exactly the genes passed as test_genes_names.

    Oracle: the mean per-gene score that compare_spatial_geneexp computes with NumPy
    from the saved mapping, restricted to the genes flagged is_test.
    """
    adata_sc, adata_st = make_toy_pair()
    monkeypatch.chdir(tmp_path)
    adata_map, model, datamodule = tgl.map_cells_to_space(
        adata_sc, adata_st, num_epochs=3, random_state=0, device="cpu", **GENE_SPLIT
    )

    test_results = tgl.test_mapping_experiment(model, datamodule)[0]

    adata_ge = tgl.project_sc_genes_onto_space(adata_map, datamodule)
    per_gene = tgl.compare_spatial_geneexp(adata_ge, datamodule)
    assert set(per_gene.index[per_gene["is_test"]]) == set(GENE_SPLIT["test_genes_names"])
    assert not (per_gene["is_test"] & (per_gene["is_training"] | per_gene["is_validation"])).any()
    assert test_results["test_score"] == pytest.approx(per_gene.loc[per_gene["is_test"], "score"].mean(), rel=1e-5)


def test_test_genes_do_not_influence_training(tmp_path, monkeypatch, make_toy_pair):
    """Held-out means held-out: changing the test genes' data must leave the mapping unchanged."""
    monkeypatch.chdir(tmp_path)
    mappings = []
    for shuffle_test_genes in (False, True):
        adata_sc, adata_st = make_toy_pair()
        if shuffle_test_genes:
            rows = np.random.default_rng(1).permutation(adata_st.n_obs)
            adata_st.X[:, 9:12] = adata_st.X[rows, 9:12]
        adata_map, _, _ = tgl.map_cells_to_space(
            adata_sc, adata_st, num_epochs=3, random_state=0, device="cpu", **GENE_SPLIT
        )
        mappings.append(adata_map.X)

    np.testing.assert_array_equal(mappings[0], mappings[1])
