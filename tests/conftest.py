"""
Shared fixtures for the tangramlit tests.

The tests run on a small synthetic dataset built here (no data files, CPU only):

    pytest

Each test checks an invariant against an independent oracle rather than against
stored numbers, so it stays valid when the training protocol changes.
"""

import anndata as ad
import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")  # validation draws a figure; never open a window in tests

N_CELLS, N_GENES, GRID = 40, 12, 5  # 5 x 5 grid -> 25 spots


def _toy_pair(seed=0):
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


@pytest.fixture(scope="session")
def make_toy_pair():
    """Builder of the toy data: each call returns a fresh (adata_sc, adata_st) pair, safe to modify."""
    return _toy_pair
