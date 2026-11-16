import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from ..mapping.lit_mapper import poly2_auc

# NOTE: plotting functions return their figure(s) and never call plt.show(), so that they
# work unchanged in notebooks, in scripts and on headless machines.


def plot_training_scores(df_g, bins=10, alpha=0.7):
    """
        Plots the 4-panel training diagnosis plot. Restricted on genes flagged as 'is_training'.

        Args:
            df_g (pandas.DataFrame): Contains overlap genes sparsity/score values produced by mapping_utils.compare_spatial_gene_expr()
            bins (int or string): Optional. Default is 10.
            alpha (float): Optional. Ranges from 0-1, and controls the opacity. Default is 0.7.

        Returns:
            matplotlib Figure.
    """
    df_g = df_g.loc[df_g['is_training']]

    fig, axs = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    axs_f = axs.flatten()

    # set limits for axis
    axs_f[0].set_ylim([0.0, 1.0])
    for i in range(1, len(axs_f)):
        axs_f[i].set_xlim([0.0, 1.0])
        axs_f[i].set_ylim([0.0, 1.0])

    axs_f[0].set_title('Training scores for single genes')
    sns.histplot(data=df_g, y="score", bins=bins, ax=axs_f[0], color="coral")

    axs_f[1].set_title("score vs sparsity (single cells)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_sc",
        ax=axs_f[1],
        alpha=alpha,
        color="coral",
    )

    axs_f[2].set_title("score vs sparsity (spatial)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_st",
        ax=axs_f[2],
        alpha=alpha,
        color="coral",
    )

    axs_f[3].set_title("score vs sparsity (sp - sc)")
    sns.scatterplot(
        data=df_g,
        y="score",
        x="sparsity_diff",
        ax=axs_f[3],
        alpha=alpha,
        color="coral",
    )

    fig.tight_layout()

    return fig


def _select_genes(df, mask_col, stat_col=None):
    """Genes flagged in `mask_col` with a non-zero score and, if given, a non-null value in `stat_col`."""
    grp = df.loc[df[mask_col] == True].copy()
    # remove zero-score entries (poly fit / auc routine requires non-zero scores)
    grp = grp.loc[grp['score'] != 0]
    if stat_col is not None:
        grp = grp.loc[~grp[stat_col].isnull()]
    return grp


def _auc_figure(df, mask_col, title):
    """Quadratic fit of spatial sparsity vs score for one gene group. Returns (figure, auc), or (None, nan) if the group is empty."""
    grp = _select_genes(df, mask_col)
    if grp.empty:
        return None, np.nan

    auc_score, ((pol_xs, pol_ys), (xs, ys)) = poly2_auc(grp['score'], grp['sparsity_st'], plot_auc=False)
    fig = plt.figure(figsize=(6, 5))
    plt.plot(pol_xs, pol_ys, c='r', label=f'poly fit')
    sns.scatterplot(x=xs, y=ys, alpha=0.6, edgecolors='face', color='blue', label='genes')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel('score')
    plt.ylabel('spatial sparsity')
    plt.tick_params(axis='both', labelsize=8)
    plt.legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(0.03, 0.1, f'AUC={np.round(auc_score,3)}', fontsize=11, verticalalignment='top', bbox=props)
    plt.title(title)

    return fig, auc_score


def plot_auc_curve(df_g, plot_train=True, plot_validation=True):
    """
        Plots the quadratic polynomial fit of spatial sparsity vs score and its AUC, separately for
        training and validation genes.

        Args:
            df_g (pandas.DataFrame): returned by compare_spatial_gene_expr(adata_ge, adata_sp).
            plot_train (bool): if True, plot genes flagged as training (df_g['is_training'] == True).
            plot_validation (bool): if True, plot genes flagged as validation (df_g['is_validation'] == True).

        Returns:
            tuple: ((fig_train, fig_val), (auc_train, auc_val)). A figure is None and its AUC is NaN
            when the group was not requested or has no genes.
        """
    # At least one of the plotting flags must be True
    if not plot_train and not plot_validation:
        raise ValueError('At least one of plot_train or plot_validation must be True.')

    fig_train, auc_train = None, np.nan
    fig_val, auc_val = None, np.nan

    if plot_train and 'is_training' in df_g.columns:
        fig_train, auc_train = _auc_figure(df_g, 'is_training', 'Training quadratic polynomial fit')
    if plot_validation and 'is_validation' in df_g.columns:
        fig_val, auc_val = _auc_figure(df_g, 'is_validation', 'Validation quadratic polynomial fit')

    if fig_train is None and fig_val is None:
        raise ValueError('No genes with non-null score in the requested groups, CosSim not computable.')

    # Return the two separate figures (either may be None)
    return (fig_train, fig_val), (auc_train, auc_val)


def _safe_corr(x, y):
    """Pearson correlation ignoring NaNs; NaN if fewer than two valid pairs."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = (~np.isnan(x)) & (~np.isnan(y))
    if mask.sum() < 2:
        return np.nan
    try:
        return float(np.corrcoef(x[mask], y[mask])[0, 1])
    except Exception:
        return np.nan


def _score_vs_stat_figure(df, mask_col, stat_col, title, plot_fit):
    """Score vs one spatial-autocorrelation statistic for one gene group. Returns (figure, correlation), or (None, nan) if the group is empty."""
    grp = _select_genes(df, mask_col, stat_col)
    if grp.empty:
        return None, np.nan

    xs = grp['score']
    ys = grp[stat_col]
    corr = _safe_corr(xs, ys)
    fig = plt.figure(figsize=(6, 5))
    sns.scatterplot(x=ys, y=xs, alpha=0.6, edgecolors='face', color='blue', label='genes')
    if plot_fit and xs.size > 1:
        mask = (~np.isnan(xs)) & (~np.isnan(ys))
        if mask.sum() > 1 and np.unique(ys[mask]).size > 1:
            coeffs = np.polyfit(ys[mask], xs[mask], 1)
            xmin = float(np.min(ys[mask]))
            xmax = float(np.max(ys[mask]))
            rng = xmax - xmin
            delta = rng * 0.05 if rng > 0 else max(0.1, abs(xmin) * 0.05)
            fit_x = np.linspace(xmin - delta, xmax + delta, 200)
            fit_y = np.polyval(coeffs, fit_x)
            plt.plot(fit_x, fit_y, c='r', label='linear fit')
    plt.ylim([0.0, 1.0])
    plt.gca().set_aspect(.5)
    plt.xlabel(stat_col)
    plt.ylabel('score')
    plt.tick_params(axis='both', labelsize=8)
    plt.legend()
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    plt.text(0.03, 0.1, f'corr={np.round(corr,3)}', fontsize=11, verticalalignment='top', bbox=props)
    plt.title(title)

    return fig, corr


def plot_score_SA_corr(df_g, plot_train=True, plot_validation=True, plot_fit=True):
    """
    Plots score vs spatial-autocorrelation statistics (Moran's I and Geary's C) and
    computes linear correlation coefficients for training/validation gene groups.

    Args:
        df_g (pandas.DataFrame): DataFrame returned by compare_spatial_gene_expr(), must contain
            columns 'score' and one or both of 'moranI' and 'gearyC' (or 'geary').
        plot_train (bool): If True, produce plots for genes flagged as training ('is_training').
        plot_validation (bool): If True, produce plots for genes flagged as validation ('is_validation').
        plot_fit (bool): If True, overlay a linear fit line on the scatter plots.

    Returns:
        tuple: (figures_tuple, correlations_tuple)
            figures_tuple = (fig_train_moran, fig_val_moran, fig_train_geary, fig_val_geary)
            correlations_tuple = (corr_train_moran, corr_val_moran, corr_train_geary, corr_val_geary)
    """
    if not plot_train and not plot_validation:
        raise ValueError('At least one of plot_train or plot_validation must be True.')

    # Determine available spatial-autocorr columns
    moran_col = 'moranI' if 'moranI' in df_g.columns else None
    geary_col = 'gearyC' if 'gearyC' in df_g.columns else ('geary' if 'geary' in df_g.columns else None)

    if moran_col is None and geary_col is None:
        raise ValueError("Neither 'moranI' nor 'gearyC'/'geary' found in dataframe.")

    # One (figure, correlation) pair per gene group and statistic; (None, nan) when not produced
    results = {(group, stat): (None, np.nan) for group in ('train', 'val') for stat in ('moran', 'geary')}

    groups = []
    if plot_train and 'is_training' in df_g.columns:
        groups.append(('train', 'is_training', 'Training'))
    if plot_validation and 'is_validation' in df_g.columns:
        groups.append(('val', 'is_validation', 'Validation'))

    for group, mask_col, group_title in groups:
        for stat, stat_col, stat_title in (('moran', moran_col, 'Moran I'), ('geary', geary_col, 'Geary C')):
            if stat_col is not None:
                results[(group, stat)] = _score_vs_stat_figure(
                    df_g, mask_col, stat_col, f'{group_title} {stat_title} vs score', plot_fit
                )

    if all(fig is None for fig, _ in results.values()):
        raise ValueError('No genes with non-null score/statistic in the requested groups, cannot compute correlations.')

    order = [('train', 'moran'), ('val', 'moran'), ('train', 'geary'), ('val', 'geary')]
    figures = tuple(results[key][0] for key in order)
    corrs = tuple(results[key][1] for key in order)
    return figures, corrs


def plot_score_histograms(df_g, bins=10, alpha=0.7, plot_train=True, plot_validation=True, remove_zero=True):
    """
    Plot histograms of `score` values for training and validation gene groups.

    Args:
        df_g (pandas.DataFrame): DataFrame containing at least 'score' and optional
            boolean columns 'is_training' and 'is_validation'. If those flags are
            missing, histograms for the corresponding group will be None.
        bins (int or sequence): Number of bins or bin edges for the histogram.
        alpha (float): Opacity for histogram bars.
        plot_train (bool): If True, produce the training histogram.
        plot_validation (bool): If True, produce the validation histogram.
        remove_zero (bool): If True, drop rows with score == 0 before plotting.

    Returns:
        tuple: (fig_train, fig_val) where each is a matplotlib Figure or None.
    """
    df_sel = df_g.copy()

    if not plot_train and not plot_validation:
        raise ValueError('At least one of plot_train or plot_validation must be True.')

    fig_train = None
    fig_val = None

    # helper to prepare series
    def get_scores(df, mask_col=None):
        if mask_col is not None and mask_col in df.columns:
            s = df.loc[df[mask_col] == True, 'score'].copy()
        else:
            s = df['score'].copy()
        if remove_zero:
            s = s.loc[s != 0]
        s = s.dropna()
        return s

    # Training histogram
    if plot_train and 'is_training' in df_sel.columns:
        scores_tr = get_scores(df_sel, 'is_training')
        if not scores_tr.empty:
            fig_train = plt.figure(figsize=(6, 4))
            sns.histplot(scores_tr, bins=bins, kde=False, color='coral', alpha=alpha)
            plt.xlim([0.0, 1.0])
            plt.ylim(bottom=0)
            plt.xlabel('score')
            plt.ylabel('count')
            plt.title('Training gene score distribution')
            plt.tight_layout()

    # Validation histogram
    if plot_validation and 'is_validation' in df_sel.columns:
        scores_val = get_scores(df_sel, 'is_validation')
        if not scores_val.empty:
            fig_val = plt.figure(figsize=(6, 4))
            sns.histplot(scores_val, bins=bins, kde=False, color='steelblue', alpha=alpha)
            plt.xlim([0.0, 1.0])
            plt.ylim(bottom=0)
            plt.xlabel('score')
            plt.ylabel('count')
            plt.title('Validation gene score distribution')
            plt.tight_layout()

    return fig_train, fig_val
