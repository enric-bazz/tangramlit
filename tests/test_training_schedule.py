"""Training schedule: learning rate scheduler, early stopping and restoring of the best mapping."""

import torch

from tangramlit.mapping import MapperLightning


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
