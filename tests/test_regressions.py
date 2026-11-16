"""
Regression tests.

They run on a small synthetic dataset built in this file (no data files, CPU only):

    pytest tests/test_regressions.py

Each test checks an invariant against an independent oracle rather than against
stored numbers, so it stays valid when the training protocol changes.
"""

import os
from types import SimpleNamespace

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest
import scipy.stats as st
import torch
import yaml

matplotlib.use("Agg")  # validation draws a figure; never open a window in tests

import tangramlit as tgl
from tangramlit.data.paired_data_module import annotate_gene_sparsity

N_CELLS, N_GENES, GRID = 40, 12, 5  # 5 x 5 grid -> 25 spots
ALL_SPATIAL_LAMBDAS = dict(
    lambda_neighborhood_g1=1,
    lambda_ct_islands=1,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1,
)


def make_toy_pair(seed=0):
    """Single-cell and spatial AnnData sharing all genes, with mixed-case gene names."""
    rng = np.random.default_rng(seed)
    genes = [f"Gene{i}" for i in range(N_GENES)]

    adata_sc = ad.AnnData(
        X=rng.poisson(2.0, size=(N_CELLS, N_GENES)).astype(np.float32),
        obs=pd.DataFrame(
            {"cell_type": rng.choice(["a", "b", "c"], size=N_CELLS)},
            index=[f"cell{i}" for i in range(N_CELLS)],
        ),
        var=pd.DataFrame(index=genes),
    )
    xx, yy = np.meshgrid(np.arange(GRID), np.arange(GRID))
    adata_st = ad.AnnData(
        X=rng.poisson(2.0, size=(GRID * GRID, N_GENES)).astype(np.float32),
        obs=pd.DataFrame(index=[f"spot{i}" for i in range(GRID * GRID)]),
        var=pd.DataFrame(index=genes),
    )
    adata_st.obsm["spatial"] = np.column_stack([xx.ravel(), yy.ravel()]).astype(float)
    return adata_sc, adata_st


@pytest.fixture(scope="module")
def mapper_with_all_spatial_terms(tmp_path_factory):
    """A mapper trained for two epochs with every spatially weighted term switched on."""
    adata_sc, adata_st = make_toy_pair()
    cwd = os.getcwd()
    os.chdir(tmp_path_factory.mktemp("mapping"))  # tb_logs/ is written to the working directory
    try:
        _, model, _ = tgl.map_cells_to_space(
            adata_sc,
            adata_st,
            cluster_label="cell_type",
            num_epochs=2,
            random_state=0,
            device="cpu",
            **ALL_SPATIAL_LAMBDAS,
        )
    finally:
        os.chdir(cwd)
    return model


def test_benchmark_metrics_belong_to_their_gene():
    """benchmark_mapping must attach each metric to the gene it was computed on.

    The input table is sorted by score, so positional assignment shuffles the metrics
    across genes. Oracle: Pearson correlation recomputed gene by gene.
    """
    _, adata_st = make_toy_pair()
    adata_sc, _ = make_toy_pair()
    for adata in (adata_sc, adata_st):
        adata.var_names = adata.var_names.str.lower()
        annotate_gene_sparsity(adata)
    adata_st.var["is_training"] = True
    datamodule = SimpleNamespace(adata_sc=adata_sc, adata_st=adata_st)

    # Any prediction works; it only has to differ from the truth gene by gene
    predicted = np.random.default_rng(1).poisson(2.0, size=adata_st.shape).astype(np.float32)
    adata_ge = ad.AnnData(X=predicted, obs=adata_st.obs.copy(), var=adata_st.var.copy())

    scores = tgl.compare_spatial_geneexp(adata_ge, datamodule)
    assert not scores.index.equals(adata_st.var_names), "toy data must exercise the sorted case"

    per_gene = tgl.benchmark_mapping(adata_ge, datamodule, scores)

    expected_pcc = pd.Series(
        [st.pearsonr(adata_st[:, g].X.ravel(), adata_ge[:, g].X.ravel())[0] for g in per_gene.index],
        index=per_gene.index,
    )
    np.testing.assert_allclose(per_gene["PCC"], expected_pcc, rtol=1e-6)


def test_spatial_weight_matrices_are_independent(mapper_with_all_spatial_terms):
    """Each spatial weight matrix keeps its own diagonal.

    The cell-type-islands filter excludes the spot itself, the Getis-Ord weights include
    it. If they share memory the islands term compares a spot with itself and is always 0.
    """
    model = mapper_with_all_spatial_terms
    assert model.neighborhood_filter.diagonal().abs().sum() == 0
    assert (model.spatial_weights_getisord.diagonal() == 1).all()
    assert model.spatial_weights_morangeary.diagonal().abs().sum() == 0


def test_lisa_references_are_registered_buffers(mapper_with_all_spatial_terms):
    """LISA reference tensors must follow the module to its device.

    They are computed in setup(), before Lightning moves the model to the accelerator.
    Only registered buffers are moved, so this is the CPU-checkable form of
    "training with LISA terms works on GPU".
    """
    buffers = dict(mapper_with_all_spatial_terms.named_buffers())
    for name in ("getis_ord_G_star_ref", "moran_I_ref", "gearys_C_ref"):
        assert name in buffers and buffers[name] is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_all_spatial_terms_train_on_gpu(tmp_path, monkeypatch):
    """The direct check behind the test above: two epochs on GPU with every term on."""
    adata_sc, adata_st = make_toy_pair()
    monkeypatch.chdir(tmp_path)
    adata_map, _, _ = tgl.map_cells_to_space(
        adata_sc,
        adata_st,
        cluster_label="cell_type",
        num_epochs=2,
        random_state=0,
        device="gpu",
        **ALL_SPATIAL_LAMBDAS,
    )
    assert len(adata_map.uns["training_history"]["total_loss"]) == 2


def test_cv_summary_aggregates_every_fold(tmp_path, monkeypatch):
    """`tangramlit cv`: the summary must be the statistics of the per-fold table.

    Runs the command end to end, so it is also the smoke test for the whole
    train -> validate -> project -> benchmark path.
    """
    from tangramlit.cli.main import main

    adata_sc, adata_st = make_toy_pair()
    adata_sc.write_h5ad(tmp_path / "sc.h5ad")
    adata_st.write_h5ad(tmp_path / "st.h5ad")
    with open(tmp_path / "config.yaml", "w") as f:
        yaml.dump({"num_epochs": 3, "random_state": 0, "lambda_g1": 1}, f)

    monkeypatch.chdir(tmp_path)
    out = tmp_path / "out"
    main(["cv", "-i", str(tmp_path), "-sc", "sc.h5ad", "-ist", "st.h5ad",
          "-c", "config.yaml", "-o", str(out), "--k-folds", "3"])

    per_fold = pd.read_csv(out / "cv_metrics_per_fold.csv")
    with open(out / "cv_metrics.yaml") as f:
        summary = yaml.safe_load(f)

    assert len(per_fold) == 3
    for metric in ("val_score", "PCC", "RMSE", "JS", "SSIM"):
        assert summary[metric]["mean"] == pytest.approx(per_fold[metric].mean())
        assert summary[metric]["std"] == pytest.approx(per_fold[metric].std())
