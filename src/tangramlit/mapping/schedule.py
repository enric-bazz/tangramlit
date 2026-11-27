"""
Training schedule shared by the mapping and the hyperparameter tuning.

Both train with the same stopping rule, validation frequency and precision, so that the loss
coefficients selected by the tuning are selected under the conditions they are later used in.
"""
import math
from dataclasses import dataclass

from lightning.pytorch.callbacks import Callback, EarlyStopping


@dataclass(frozen=True)
class TrainingSchedule:
    """
    When to validate, when to stop and which mapping to keep. All durations are in epochs.

    Args:
        num_epochs (int): Maximum number of epochs. Default is 500.
        warmup_epochs (int): Epochs during which early stopping is not evaluated. Default is 50.
        patience (int): Epochs without improvement of `monitor` after which training stops. The earliest
            possible stop is therefore at `warmup_epochs + patience` epochs. Default is 30.
        min_delta (float): Minimum increase of `monitor` that counts as an improvement. Default is 0.001.
        monitor (str): Validation metric that drives early stopping, the choice of the best mapping and the
            hyperparameter tuning. Higher is better. Default is 'val_objective' (see MapperLightning._evaluation_step()),
            'val_score' is the gene-voxel term alone.
        check_val_every_n_epoch (int): Validation frequency. Default is 1.
        restore_best (bool): Return the mapping of the epoch with the best `monitor` rather than the last one.
            Keeps one extra copy of the mapping matrix on the training device. Default is True.
    """
    num_epochs: int = 500
    warmup_epochs: int = 50
    patience: int = 30
    min_delta: float = 0.001
    monitor: str = "val_objective"
    check_val_every_n_epoch: int = 1
    restore_best: bool = True

    def __post_init__(self):
        if self.num_epochs <= 0:
            raise ValueError("num_epochs must be positive")
        if self.warmup_epochs < 0 or self.patience <= 0 or self.check_val_every_n_epoch <= 0:
            raise ValueError("warmup_epochs must be >= 0, patience and check_val_every_n_epoch must be positive")

    @property
    def can_stop_early(self):
        """Whether `num_epochs` leaves room for early stopping to act."""
        return self.num_epochs > self.warmup_epochs + self.patience

    def callbacks(self):
        """
        Build new callbacks for one training run: early stopping and the tracker of the best mapping.
        The tracker is always the last one of the list.
        """
        early_stop = WarmupEarlyStopping(
            warmup_epochs=self.warmup_epochs,
            monitor=self.monitor,
            min_delta=self.min_delta,
            # EarlyStopping counts validation checks: convert the patience from epochs
            patience=math.ceil(self.patience / self.check_val_every_n_epoch),
            mode="max",
            verbose=False,
            check_on_train_epoch_end=False,  # the monitored metric is logged in validation_step()
        )
        best_state = BestStateTracker(monitor=self.monitor, restore=self.restore_best)
        return [early_stop, best_state]

    def trainer_kwargs(self):
        """Trainer arguments that must be the same in every training run."""
        return dict(
            max_epochs=self.num_epochs,
            check_val_every_n_epoch=self.check_val_every_n_epoch,
            log_every_n_steps=1,  # one training step per epoch
            enable_checkpointing=False,  # the best mapping is kept in memory by BestStateTracker
            precision="32-true",  # log() and softmax() of the mapping matrix are not safe in half precision
        )


class WarmupEarlyStopping(EarlyStopping):
    """
    Early stopping that is not evaluated during the first `warmup_epochs` epochs.

    `Trainer(min_epochs=...)` is not equivalent: the stop request raised before `min_epochs` is kept,
    so training stops exactly at `min_epochs` even if the monitored metric has improved in the meantime.
    """

    def __init__(self, warmup_epochs=0, **kwargs):
        super().__init__(**kwargs)
        self.warmup_epochs = warmup_epochs

    def on_validation_end(self, trainer, pl_module):
        # trainer.current_epoch is the 0-based index of the epoch just trained
        if trainer.current_epoch + 1 < self.warmup_epochs:
            return
        super().on_validation_end(trainer, pl_module)


class BestStateTracker(Callback):
    """
    Track the best value of the monitored validation metric (higher is better) and the epoch it was reached at.
    If `restore` is True, also keep a copy of the model parameters at that epoch and load it back when training ends.

    Attributes:
        best_value (float): best value of the monitored metric, None before the first validation.
        best_epoch (int): 0-based epoch of the best value.
    """

    def __init__(self, monitor, restore=True):
        super().__init__()
        self.monitor = monitor
        self.restore = restore
        self.best_value = None
        self.best_epoch = None
        self._best_state = None

    def on_validation_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return
        if self.monitor not in trainer.callback_metrics:
            raise RuntimeError(
                f"Monitored metric '{self.monitor}' not found among the validation metrics: "
                f"{sorted(k for k in trainer.callback_metrics if k.startswith('val_'))}"
            )
        current = float(trainer.callback_metrics[self.monitor])
        if not math.isfinite(current):
            return
        if self.best_value is None or current > self.best_value:
            self.best_value = current
            self.best_epoch = trainer.current_epoch
            if self.restore:
                state = pl_module.state_dict()
                if self._best_state is None:
                    self._best_state = {name: tensor.detach().clone() for name, tensor in state.items()}
                else:
                    for name, tensor in state.items():
                        self._best_state[name].copy_(tensor)  # reuse the allocated copy

    def on_train_end(self, trainer, pl_module):
        if self.restore and self._best_state is not None:
            pl_module.load_state_dict(self._best_state)
