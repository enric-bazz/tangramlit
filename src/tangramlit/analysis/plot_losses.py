from matplotlib import pyplot as plt
import numpy as np
import torch
import math

# NOTE: plotting functions return their figure(s) and never call plt.show(), so that they
# work unchanged in notebooks, in scripts and on headless machines.


def _history_to_array(values):
    """Turn a history entry (list of floats or scalar tensors, or an array after an h5ad round trip) into a float array."""
    return np.asarray(
        [v.detach().cpu().item() if torch.is_tensor(v) else v for v in values],
        dtype=float,
    )


def plot_training_history(adata_map, hyperpams=None, lambda_scale=True, log_scale=False, show_total_loss=False):
    """
        Plots a panel with all loss term curves in training.

        Args:
            adata_map (anndata object): input containing .uns["training_history"] returned by map_cells_to_space()
            hyperpams (dict): dictionary containing the hyperparameters used for the mapping
            lambda_scale (bool): Whether to scale the loss terms by lambda (default: True)
            log_scale (bool): Whether the y axis plots should be in log-scale (default: False)
            show_total_loss (bool): Whether to show the total loss term named 'loss' (default: False)

        Returns:
            matplotlib Figure.
            Note that the trainig step stores in adata_map.uns["training_history"] the loss terms for each epoch already scaled
            by their respective hyperparameters, thus to get non scaled values we divide by the respective lambda.
    """

    # Check if training history is present
    if not "training_history" in adata_map.uns.keys():
        raise ValueError("Missing training history in mapped input data.")

    if not lambda_scale and hyperpams is None:
        raise ValueError("Missing hyperparamters for re-scaling.")

    # Keep the loss terms that were recorded (terms with lambda = 0 have an empty history)
    loss_dict = {
        k: _history_to_array(v)
        for k, v in adata_map.uns["training_history"].items()
        if len(v) > 0
    }

    # Scale by lambda (bool)
    loss_lambda_map = {
        "main_loss": "lambda_g1",
        "vg_reg": "lambda_g2",
        "kl_reg": "lambda_d",
        "entropy_reg": "lambda_r",
        "l1_term": "lambda_l1",
        "l2_term": "lambda_l2",
        "count_reg": "lambda_count",
        "filt_reg": "lambda_f_reg",
        "sparsity_term": "lambda_sparsity_g1",
        "neighborhood_term": "lambda_neighborhood_g1",
        "ct_island_term": "lambda_ct_islands",
        "getis_ord_term": "lambda_getis_ord",
        "moran_term": "lambda_moran",
        "geary_term": "lambda_geary",
    }
    if not lambda_scale:
        for loss_key in (loss_dict.keys() & loss_lambda_map.keys()):
            loss_dict[loss_key] = loss_dict[loss_key] / hyperpams[loss_lambda_map[loss_key]]

    # Optionally hide the total loss term named 'total_loss'
    if not show_total_loss and 'total_loss' in loss_dict:
        # remove from dict so it's not plotted below
        loss_dict.pop('total_loss', None)

    # Create plot
    fig = plt.figure(figsize=(10,20))

    title = 'Loss terms over epochs'
    if lambda_scale:
        title += ' (scaled by lambda)'
    if log_scale:
        title = title + ' (logscale)'

    for curve in loss_dict:
        if log_scale:
            plt.semilogy(abs(loss_dict[curve]), label=curve)
        else:
            plt.plot(loss_dict[curve], label=curve)
        plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(title)

    return fig


def plot_loss_terms(
    adata_map,
    loss_key,
    lambda_coeff=None,
    lambda_scale=False,
    log_scale=False,
    make_subplot=False,
    subplot_shape=None,
):
    """
    Plot one or more loss terms from adata_map.uns['training_history'].

    Args:
        adata_map (AnnData): contains .uns['training_history']
        loss_key (str | list[str]): one or more keys of loss terms
        lambda_coeff (float | list[float] | None): λ coefficient(s)
        lambda_scale (bool): if True, plot λ-weighted loss (as stored);
                             if False, divide by λ to show unweighted loss
        log_scale (bool): if True, use semilog-y
        make_subplot (bool): if True, draw all curves in subplots
        subplot_shape (tuple[int, int] | None): (nrows, ncols) for subplots

    Returns:
        A single matplotlib Figure if make_subplot is True, otherwise a list with one Figure per loss term.
    """
    history = adata_map.uns.get("training_history")
    if history is None:
        raise ValueError("Missing 'training_history' in adata_map.uns.")

    # Normalize inputs
    if isinstance(loss_key, str):
        loss_key = [loss_key]
    if lambda_coeff is None:
        lambda_coeff = [1.0] * len(loss_key)
    elif isinstance(lambda_coeff, (float, int)):
        lambda_coeff = [lambda_coeff]
    if len(loss_key) != len(lambda_coeff):
        raise ValueError("loss_key and lambda_coeff must have equal length.")

    n_terms = len(loss_key)
    figures = []

    if make_subplot:
        if subplot_shape is None:  # square subplot shape
            ncols = math.ceil(math.sqrt(n_terms))
            nrows = math.ceil(n_terms / ncols)
        else:
            nrows, ncols = subplot_shape
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.array(axes).reshape(-1)
        figures.append(fig)

    for i, (key, coeff) in enumerate(zip(loss_key, lambda_coeff)):
        if key not in history:
            raise ValueError(f"Loss term '{key}' not found in training_history.")
        if len(history[key]) == 0:
            raise ValueError(f"Loss term '{key}' has empty history.")
        values = _history_to_array(history[key])

        # Unscale if required
        if not lambda_scale and coeff != 0:
            values = values / coeff

        # pick axis
        if make_subplot:
            ax = axes[i]
        else:
            fig = plt.figure(figsize=(7, 4))
            ax = plt.gca()
            figures.append(fig)

        if log_scale:
            ax.semilogy(np.abs(values))
        else:
            ax.plot(values)

        ax.set_title(f"{key} ({'λ-scaled' if lambda_scale else 'unscaled'})")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")

    for fig in figures:
        fig.tight_layout()

    return figures[0] if make_subplot else figures


def plot_validation_metrics_history(adata_map, figsize=(10, 5), add_training_scores=False, hyperpams=None):
    """
    Plots the history of validation metrics stored in adata_map.uns['validation_history'] in separated plots.
    Optionally compare training score and sparsity-weighted score with the validation ones if add_training_scores is True.

    Args:
        adata_map: anndata object returned by the mapping containing the validation history
        figsize: tuple specifying the figure size
        add_training_scores (bool): overlay the training score on 'val_score' and the training sparsity-weighted
            score on 'val_sparsity-weighted_score'. A training curve is skipped when its loss term was not active.
        hyperpams (dict): Optional. Hyperparameters used for the mapping (e.g. mapper.hparams). The training history
            stores lambda-weighted terms while validation scores are unweighted: when given, training curves are
            divided by their lambda to make the two comparable; otherwise they are plotted as stored and labelled so.

    Returns:
        dict mapping each validation metric name to its matplotlib Figure.
    """
    if 'validation_history' not in adata_map.uns:
        raise ValueError("Missing 'validation_history' in adata_map.uns.")

    val_history = adata_map.uns['validation_history']
    training_history = adata_map.uns.get('training_history', {})

    # Validation metric -> (training loss term, its lambda, label)
    training_counterpart = {
        'val_score': ('main_loss', 'lambda_g1', 'Score'),
        'val_sparsity-weighted_score': ('sparsity_term', 'lambda_sparsity_g1', 'Sparsity-Weighted Score'),
    }

    figures = {}
    for metric in val_history.keys():
        val_values = _history_to_array(val_history[metric])
        val_epochs = np.arange(len(val_values))

        train_values = None
        if add_training_scores and metric in training_counterpart:
            term, lambda_name, label = training_counterpart[metric]
            if len(training_history.get(term, [])) > 0:
                train_values = _history_to_array(training_history[term])
                train_label = f'Training {label}'
                if hyperpams is not None:
                    train_values = train_values / hyperpams[lambda_name]
                else:
                    train_label += f' (x {lambda_name})'
                # Place validation points on the epochs they were computed at (validation every N epochs)
                if len(val_values) > 0:
                    every = max(1, round(len(train_values) / len(val_values)))
                    val_epochs = (val_epochs + 1) * every - 1

        fig = plt.figure(figsize=figsize)
        if train_values is not None:
            plt.plot(np.arange(len(train_values)), train_values, '-o', label=train_label)
            plt.plot(val_epochs, val_values, '-o', label=f'Validation {label}')
            plt.ylabel(label)
            plt.title(f'Training vs Validation {label} over epochs')
            plt.legend()
        else:
            plt.plot(val_epochs, val_values, '-o')
            plt.ylabel(metric)
            plt.title(f'Validation {metric} over epochs')

        plt.xlabel('Epoch')
        plt.grid(True)
        figures[metric] = fig

    return figures
