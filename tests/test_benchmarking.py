"""Benchmarking metrics (PCC, RMSE, JS, SSIM) added to the per-gene table."""

from types import SimpleNamespace

import anndata as ad
import numpy as np
import pandas as pd
import scipy.stats as st

import tangramlit as tgl
from tangramlit.data.paired_data_module import annotate_gene_sparsity


def test_benchmark_metrics_belong_to_their_gene(make_toy_pair):
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
