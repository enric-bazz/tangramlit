"""Command-line interface, run end to end on files written to a temporary directory."""

import pandas as pd
import pytest
import yaml


def test_cv_summary_aggregates_every_fold(tmp_path, monkeypatch, make_toy_pair):
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
