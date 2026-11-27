"""Training schedule: learning rate scheduler, early stopping and restoring of the best mapping."""

import pytest
import torch
from lightning.pytorch import LightningModule, Trainer
from torch.utils.data import DataLoader, TensorDataset

import tangramlit as tgl
from tangramlit.mapping import MapperLightning
from tangramlit.mapping.schedule import TrainingSchedule

GENE_SPLIT = dict(
    train_genes_names=[f"gene{i}" for i in range(0, 6)],
    val_genes_names=[f"gene{i}" for i in range(6, 9)],
    test_genes_names=[f"gene{i}" for i in range(9, 12)],
)


def test_lr_is_reduced_on_a_plateau_of_negative_loss():
    """The learning rate must drop when a negative loss stops improving.

    The Tangram loss is negative as soon as the similarity terms dominate. A relative
    plateau threshold then accepts every loss close to the best one as an improvement.
    """
    model = MapperLightning()
    model.M = torch.nn.Parameter(torch.zeros(2, 2))
    [optimizer], [scheduler_config] = model.configure_optimizers()
    scheduler = scheduler_config["scheduler"]

    for _ in range(scheduler.patience + 2):
        optimizer.step()
        scheduler.step(-5.0)

    assert optimizer.param_groups[0]["lr"] < model.hparams.learning_rate


# --- Early stopping and best mapping, on a scripted validation curve --------------------------------

class _ScriptedModule(LightningModule):
    """A model whose validation metric follows a given curve and whose weight records the epoch."""

    def __init__(self, curve):
        super().__init__()
        self.curve = curve
        self.weight = torch.nn.Parameter(torch.zeros(1))

    def on_train_epoch_start(self):
        with torch.no_grad():
            self.weight.fill_(float(self.current_epoch))

    def training_step(self, batch):
        return (self.weight * 0).sum()

    def validation_step(self, batch):
        self.log("val_objective", self.curve[min(self.current_epoch, len(self.curve) - 1)])

    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=0.0)

    def train_dataloader(self):
        return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)

    def val_dataloader(self):
        return DataLoader(TensorDataset(torch.zeros(1, 1)), batch_size=1)


def _fit_scripted(curve, tmp_path, **schedule_kwargs):
    schedule = TrainingSchedule(**schedule_kwargs)
    callbacks = schedule.callbacks()
    model = _ScriptedModule(curve)
    trainer = Trainer(
        callbacks=callbacks, logger=False, enable_progress_bar=False, enable_model_summary=False,
        accelerator="cpu", default_root_dir=tmp_path, **schedule.trainer_kwargs(),
    )
    trainer.fit(model)
    return trainer, model, callbacks[-1]


def test_stop_request_raised_during_warmup_is_not_kept(tmp_path):
    """A metric that stalls during the warm-up and improves after it must not stop training at the warm-up.

    Curve: stalls from epoch 3 (patience runs out well before the warm-up ends), then rises again
    from epoch 10 to epoch 19, then is flat. With warm-up 10 and patience 3 the only legitimate stop
    is 3 epochs after the last improvement, i.e. after 23 epochs.
    """
    curve = [0.0, 0.1, 0.2] + [0.2] * 7 + [0.3 + 0.1 * i for i in range(10)] + [1.2] * 30
    trainer, _, _ = _fit_scripted(curve, tmp_path, num_epochs=50, warmup_epochs=10, patience=3, min_delta=0.001)

    assert trainer.current_epoch == 23


def test_no_early_stop_before_warmup_plus_patience(tmp_path):
    """A metric that never improves stops training exactly at warmup_epochs + patience epochs."""
    trainer, _, _ = _fit_scripted([1.0] * 50, tmp_path, num_epochs=50, warmup_epochs=10, patience=5)

    assert trainer.current_epoch == 15


def test_patience_is_in_epochs_whatever_the_validation_frequency(tmp_path):
    """Validating every 4 epochs must not multiply the patience by 4."""
    trainer, _, _ = _fit_scripted(
        [1.0] * 100, tmp_path, num_epochs=100, warmup_epochs=8, patience=8, check_val_every_n_epoch=4
    )

    assert trainer.current_epoch == 16


def test_best_weights_are_restored_even_if_the_peak_is_in_the_warmup(tmp_path):
    """The returned weights are those of the best epoch, not of the last one.

    The scripted model stores the epoch index in its weight, the curve peaks at epoch 2.
    """
    curve = [0.0, 0.5, 0.9, 0.4] + [0.3] * 40
    trainer, model, best = _fit_scripted(curve, tmp_path, num_epochs=40, warmup_epochs=10, patience=3)

    assert trainer.current_epoch > 3  # training went on past the peak
    assert (best.best_epoch, best.best_value) == (2, pytest.approx(0.9))
    assert model.weight.item() == 2.0


def test_last_weights_are_kept_without_restore_best(tmp_path):
    curve = [0.0, 0.5, 0.9, 0.4] + [0.3] * 40
    trainer, model, best = _fit_scripted(curve, tmp_path, num_epochs=40, warmup_epochs=10, patience=3, restore_best=False)

    assert best.best_epoch == 2
    assert model.weight.item() == trainer.current_epoch - 1


# --- The same behaviour through map_cells_to_space ---------------------------------------------------

def test_mapping_returned_is_the_one_with_the_best_monitored_value(tmp_path, monkeypatch, make_toy_pair):
    """map_cells_to_space must return the mapping of the best validation epoch.

    Oracle: validating the returned model again reproduces the maximum of the validation history.
    The toy genes are independent noise, so validation peaks early and the last epoch is not the best.
    """
    adata_sc, adata_st = make_toy_pair()
    monkeypatch.chdir(tmp_path)
    adata_map, model, datamodule = tgl.map_cells_to_space(
        adata_sc, adata_st, num_epochs=40, random_state=0, device="cpu", warmup_epochs=0, patience=40, **GENE_SPLIT
    )

    history = adata_map.uns["validation_history"]["val_objective"]
    outcome = adata_map.uns["training_schedule"]
    assert max(history) > history[-1] + 1e-4, "toy data must exercise the case best != last"
    assert outcome["best_value"] == pytest.approx(max(history))
    assert outcome["best_epoch"] == history.index(max(history))

    revalidated = tgl.validate_mapping_experiment(model, datamodule)[0]
    assert revalidated["val_objective"] == pytest.approx(max(history), rel=1e-5)
