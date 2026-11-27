"""
Hyperparameter tuning functions for Tangram using the Lightning framework.
"""
import optuna
import torch
import lightning.pytorch as lp
from optuna.integration.pytorch_lightning import PyTorchLightningPruningCallback
import gc

from tangramlit.data import PairedDataModule
from tangramlit.mapping import MapperLightning
from tangramlit.mapping.trainer import validate_mapping_inputs
from tangramlit.mapping.schedule import TrainingSchedule

def tune_loss_coefficients(
        adata_sc,                                       ######### tangram args
        adata_st,
        input_genes=None,
        train_genes_names=None,
        val_genes_names=None,
        cluster_label=None,
        learning_rate=0.1,
        num_epochs=500,
        random_state=None,
        lambda_d=1,
        lambda_g1=1,
        lambda_g2=1,
        lambda_r=1,
        lambda_l1=1,
        lambda_l2=1,
        lambda_sparsity_g1=0,
        lambda_neighborhood_g1=1,
        lambda_getis_ord=1,
        lambda_moran=1,
        lambda_geary=1,
        lambda_ct_islands=1,
        filter=True,
        target_count=None,
        lambda_count=1e-5,
        lambda_f_reg=1e-5,  
        warmup_epochs=50,                             ######### training schedule args
        patience=30,
        min_delta=0.001,
        monitor="val_objective",
        study_name="loss_coefficients_tuning",        ########### optuna args
        storage=None,  # None: in-memory study. Pass a URL (e.g. "sqlite:///study.db") to persist or resume it
        n_trials=40,
        timeout=None,
        resume=False,
        sampler_type='tpe',  # 'random' or 'tpe'
):
    """
    Tune the loss coefficients (lambdas) with Optuna.

    Every trial trains a mapping with the same schedule as map_cells_to_space() (see TrainingSchedule): same maximum
    number of epochs, warm-up, patience, validation frequency and precision. The value of a trial is the best value of
    `monitor` reached during its training, that is the value of the mapping that map_cells_to_space() would return.

    The coefficients with a value > 0 are tuned, those set to 0 stay off; lambda_g1, lambda_sparsity_g1 and the filter
    coefficients are kept at the value given.

    Args:
        adata_sc, adata_st, input_genes, train_genes_names, val_genes_names, cluster_label, learning_rate, num_epochs,
            random_state, lambda_*, filter, target_count: same as map_cells_to_space().
        warmup_epochs, patience, min_delta, monitor: training schedule, same as map_cells_to_space().
            `monitor` is the metric the study maximizes. Default is 'val_objective': the mean of the active similarity
            terms on the validation genes, computed without their lambdas so that trials are comparable.
            'val_score' selects on the gene-voxel term alone.
        study_name (str): Optuna study name.
        storage (str): Optuna storage URL. None keeps the study in memory.
        n_trials (int): number of trials.
        timeout (float): stop the study after this number of seconds.
        resume (bool): continue an existing study with the same name in `storage`.
        sampler_type (str): 'tpe' or 'random'.

    Returns:
        best_value, best_params, trials dataframe.
    """

    # Input control function
    hyperparameters = validate_mapping_inputs(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=input_genes,
        train_genes_names=train_genes_names,
        val_genes_names=val_genes_names,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        random_state=random_state,
        lambda_d=lambda_d,
        lambda_g1=lambda_g1,
        lambda_g2=lambda_g2,
        lambda_r=lambda_r,
        lambda_l1=lambda_l1,
        lambda_l2=lambda_l2,
        lambda_sparsity_g1=lambda_sparsity_g1,
        lambda_neighborhood_g1=lambda_neighborhood_g1,
        lambda_getis_ord=lambda_getis_ord,
        lambda_moran=lambda_moran,
        lambda_geary=lambda_geary,
        lambda_ct_islands=lambda_ct_islands,
        cluster_label=cluster_label,
        filter=filter,
        lambda_count=lambda_count,
        lambda_f_reg=lambda_f_reg,
        target_count=target_count,
        )

    # Training schedule of every trial: the same object map_cells_to_space() trains with
    schedule = TrainingSchedule(
        num_epochs=num_epochs,
        warmup_epochs=warmup_epochs,
        patience=patience,
        min_delta=min_delta,
        monitor=monitor,
        restore_best=False,  # only the best value is needed, not the mapping
    )
  
    # Create datamodule
    datamodule = PairedDataModule(
        adata_sc=adata_sc,
        adata_st=adata_st,
        input_genes=input_genes,
        train_genes_names=train_genes_names,
        val_genes_names=val_genes_names,
        cluster_label=cluster_label,
        )
    
    # Set up Optuna sampler and pruner
    if sampler_type == 'random':
        sampler = optuna.samplers.RandomSampler(seed=random_state)
    elif sampler_type == 'tpe':
        sampler = optuna.samplers.TPESampler(
            n_startup_trials=5,
            seed=random_state,
            multivariate=True,  # joint sampling of correlated params
            group=True,  # groups correlated parameters
            constant_liar=True,  # for parallel computing
        )
    else:
        raise ValueError("Invalid sampler type. Valid typer are 'random' and 'tpe'")
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=40, interval_steps=10)  # 5, 40, 10

    # Define or load optuna study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="maximize",
        load_if_exists=resume,
    )

    # Set info logging
    optuna.logging.set_verbosity(optuna.logging.INFO)

    # Optimize
    try:
        study.optimize(lambda t: objective(trial=t, datamodule=datamodule, training_config=hyperparameters,
                                           schedule=schedule),
                       n_trials=n_trials,
                       timeout=timeout)
    except KeyboardInterrupt:
        print("Optimization interrupted by user. Saving study.")

    print("Best trial:")
    print(f"  Value: {study.best_value}")
    print("  Params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # clean memory
    del datamodule
    gc.collect()

    return study.best_value, study.best_params, study.trials_dataframe()

    
    


def objective(
    trial: optuna.trial.Trial, 
    datamodule: PairedDataModule, 
    training_config: dict = {},
    schedule: TrainingSchedule = TrainingSchedule(restore_best=False),
):
    """
    Train one mapping with the lambdas sampled by the trial and return the best value of `schedule.monitor`.
    """
    # Work on a copy: the sampled lambdas must not leak into the next trial
    training_config = dict(training_config)

    if training_config.get('random_state') is not None:
        seed = trial.number + training_config['random_state']
        lp.seed_everything(seed, workers=True)

    # Lambda coefficient sampling

    # lambda_d
    if training_config.get('lambda_d', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_d', 1e-2, 10.0, log=True)
        training_config['lambda_d'] = lambda_coeff_val

    # lambda_g2
    if training_config.get('lambda_g2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_g2', 0, 1.0, log=False)
        training_config['lambda_g2'] = lambda_coeff_val

    # lambda_r
    if training_config.get('lambda_r', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_r', 1e-12, 1e-4, log=True)
        training_config['lambda_r'] = lambda_coeff_val

    # lambda_l1
    if training_config.get('lambda_l1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l1', 1e-18, 1e-7, log=True)
        training_config['lambda_l1'] = lambda_coeff_val

    # lambda_l2
    if training_config.get('lambda_l2', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_l2', 1e-21, 1e-7, log=True)
        training_config['lambda_l2'] = lambda_coeff_val

    # lambda_neighborhood_g1
    if training_config.get('lambda_neighborhood_g1', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_neighborhood_g1', 0, 1.0, log=False)
        training_config['lambda_neighborhood_g1'] = lambda_coeff_val

    # lambda_getis_ord
    if training_config.get('lambda_getis_ord', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_getis_ord', 0, 1.0, log=False)
        training_config['lambda_getis_ord'] = lambda_coeff_val

    # lambda_moran
    if training_config.get('lambda_moran', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_moran', 0, 1.0, log=False)
        training_config['lambda_moran'] = lambda_coeff_val

    # lambda_geary
    if training_config.get('lambda_geary', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_geary', 0, 1.0, log=False)
        training_config['lambda_geary'] = lambda_coeff_val

    # lambda_ct_islands
    if training_config.get('lambda_ct_islands', 0) > 0:
        lambda_coeff_val = trial.suggest_float('lambda_ct_islands', 1e-3, 1e3, log=True)
        training_config['lambda_ct_islands'] = lambda_coeff_val

    # Init mapper with lambdas as sampled by optuna
    model = MapperLightning(**training_config)

    # Callbacks: the shared schedule (early stopping, best value tracker) and Optuna pruning on the same metric
    schedule_callbacks = schedule.callbacks()
    best_state = schedule_callbacks[-1]
    pruning_cb = PyTorchLightningPruningCallback(trial=trial, monitor=schedule.monitor)

    # Initialize trainer with automatic device detection
    trainer = lp.Trainer(
        callbacks=[*schedule_callbacks, pruning_cb],
        enable_progress_bar=False,
        accelerator="auto",  # Automatically detect GPU/CPU
        devices="auto" if not torch.cuda.is_available() else [0],  # Use single GPU per trial if available
        **schedule.trainer_kwargs(),  # epochs, validation frequency, precision
    )

    # Fit model (trainer will call datamodule.prepare_data/setup as needed too)
    trainer.fit(model, datamodule=datamodule)

    # Value of the trial: best monitored value, the one of the mapping that map_cells_to_space() returns
    best_value = best_state.best_value
    if best_value is None:
        # Let Optuna mark the trial as failed
        raise RuntimeError(f"Monitored metric '{schedule.monitor}' was never computed during the trial.")

    # clean memory to avoid in-study build-up
    del pruning_cb
    del trainer
    del model
    gc.collect()

    return best_value