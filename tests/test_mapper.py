"""MapperLightning: spatial weight matrices and device placement of its tensors."""

import os

import pytest
import torch

import tangramlit as tgl

ALL_SPATIAL_LAMBDAS = dict(
    lambda_neighborhood_g1=1,
    lambda_ct_islands=1,
    lambda_getis_ord=1,
    lambda_moran=1,
    lambda_geary=1,
)


@pytest.fixture(scope="module")
def mapper_with_all_spatial_terms(tmp_path_factory, make_toy_pair):
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
def test_all_spatial_terms_train_on_gpu(tmp_path, monkeypatch, make_toy_pair):
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
