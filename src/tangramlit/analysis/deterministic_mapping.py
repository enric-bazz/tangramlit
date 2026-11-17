import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

### DETERMINISTIC MAPPING EVALUATION ###

# NOTE: Tangram performs an asymmetric cell to spot mapping in the sense that it estimates the probability distribution of
# each cell over spots, but conversely computes just a soft assignment/cell composition of each spot.
# This means that one hot encoding with max() over the rows corresponds to a deterministic mapping of each cell onto a single
# spot (based on its probability density function), while the same over the columns translates into a hard assignment
# of a cell for each spot that is not based on a truly probability distribution, but rather a composition vector.

def get_cell_spot_pair(adata_map, filter=False, threshold=0.5):
    """
    Perform deterministic mapping of cells to spots from the cells persepctive, i.e. each cell distribution is collapsed
    into the max value to deterministically map it into a spot.

    Args:
        adata_map: Mapping object output of map_cells_to_space
        filter (bool): Whether cell filtering is performed. Default: False.
        threshold (float): Threshold value for cell filtering.
        NOTE: Does not guarantee that all spots are assigned to at leas one cell (not feasible for spatial reconstruction).

    Returns:
        Dataframe with shape (n_cells/n_filtered_cells, 2): each row = (spot index, cell index).
    """
    # Input check
    if filter and "F_out" not in adata_map.obs.keys():
        raise ValueError("Missing final filter in mapped input data.")

    ### Deterministic mapping
    # Store mapping matrix
    mapping_matrix = adata_map.X  # shape (n_cells, n_spots)
    if not isinstance(mapping_matrix, np.ndarray):
        mapping_matrix = mapping_matrix.toarray()  # in case it's sparse
    if filter:
        # Filter out non-selected cells
        deterministic_filter = adata_map.obs['F_out'] > threshold  # shape (n_cells,)
        mapping_matrix = mapping_matrix[deterministic_filter, :]  # shape (n_filtered_cells, n_spots)
        cell_index = adata_map.obs_names[deterministic_filter]  # keep only filtered cell names
    else:
        cell_index = adata_map.obs_names
    # Collapse cell mappings to spot corresponding to max probability
    spot_number = np.argmax(mapping_matrix, axis=1)  # shape (n_cells,)
    spot_index = adata_map.var_names[spot_number]  # shape (n_cells,)

    # cell-spot pairs dataframe
    pairs_df = pd.DataFrame({
        "spot index": spot_index,
        "cell index": cell_index,
    }).reset_index(drop=True)

    return pairs_df

def get_spot_cell_pair(adata_map, filter=False, threshold=0.5):
    """
    Perform deterministic mapping of cells to spots from the spots perspective, i.e. each spot get "deterministically"
    assigned to the cell with the largest probability over it, aka its main component.

    Args:
        adata_map: Mapping object output of map_cells_to_space
        filter (bool): Whether cell filtering is performed. Default: False.
        threshold (float): Threshold value for cell filtering.

    Returns:
        DataFrame with shape (n_spots, 2): each row = (spot index, cell index).
    """
    # Input check
    if filter and "F_out" not in adata_map.obs.keys():
        raise ValueError("Missing final filter in mapped input data.")

    # Store mapping matrix
    mapping_matrix = adata_map.X  # shape (n_cells, n_spots)
    if not isinstance(mapping_matrix, np.ndarray):
        mapping_matrix = mapping_matrix.toarray()  # in case it's sparse

    # Handle cell filtering
    if filter:
        deterministic_filter = adata_map.obs['F_out'] > threshold  # shape (n_cells,)
        mapping_matrix = mapping_matrix[deterministic_filter, :]  # shape (n_filtered_cells, n_spots)
        obs_names = adata_map.obs_names[deterministic_filter]  # filtered cell names
    else:
        obs_names = adata_map.obs_names

    # Collapse spot mappings: for each spot (column), find cell with max probability
    cell_number = np.argmax(mapping_matrix, axis=0)  # shape (n_spots,)
    cell_index = obs_names[cell_number]  # map indices to cell names
    spot_index = adata_map.var_names  # all spots (always length n_spots)

    # Spot-cell pairs dataframe
    pairs_df = pd.DataFrame({
        "spot index": spot_index,
        "cell index": cell_index
    }).reset_index(drop=True)

    return pairs_df

def count_deterministic_mapping_matches(cells_to_spots_df, spots_to_cells_df):
    """
    Count the number of matches between a cells to spot deterministic mapping and a spots to cells deterministic mapping.
    This is implemented from the spots perspective. i.e. for each spot:
    - get the assigned cell id from spots_ti_cells_df
    - find the cell in cells_to_spots_df with the same cell id
    - get the assigned spot id from cells_to_spots_df
    - if it matches the spot id, increment the counter

    Args:
        cells_to_spots_df: DataFrame with shape (n_cells, 2) containing the cell index and the corresponding spot index. Output of get_cell_spot_pair().
        spots_to_cells_df: DataFrame with shape (n_spots, 2) containing the spot index and the corresponding cell index. Output of get_spot_cell_pair().

    Returns:
        The fraction of spots that match the reverse mapping.
        A spot whose cell is missing from cells_to_spots_df (e.g. the two tables were built with different
        filter settings) counts as not matching.
    """
    # Spot assigned to each cell from the cells perspective (each cell appears once)
    cell_to_spot = cells_to_spots_df.set_index('cell index')['spot index']
    # Spot that the cell assigned to each spot maps back to (NaN if the cell is missing)
    spot_idx_matched = spots_to_cells_df['cell index'].map(cell_to_spot)

    # Count number of matches
    n_matches = int((spot_idx_matched == spots_to_cells_df['spot index']).sum())

    # Compute fraction of spots that match the reverse mapping
    matches_fraction = n_matches / len(spots_to_cells_df)

    print(f">>> Fraction of spots that match the reverse mapping: {matches_fraction:.2f}")

    return matches_fraction


# Compute the quality of deterministic mapping in two ways:
#         1. For non annotated spatial data: computes gene expression similarity between deterministic cell-spot pair.
#         2. For annotated spatial data: computes cell type (annotation) accuracy between deterministic cell-spot pair.
# Each is conditioned on filtering.

def deterministic_mapping_similarity(adata_map, adata_sc, adata_st, flavour: str, filter=False, threshold=0.5, plot=True):
    """
    Compute cosine similarity between the expression profiles, limited to the shared genes, of the deterministic cell-spot pair.

    Args:
        adata_map: Mapping object output of map_cells_to_space
        adata_sc: input single cell data
        adata_st: input spatial data
        flavour (str): Either "spot_to_cell" or "cell_to_spot" for the mapping perspective.
        filter (bool): Whether cell filtering is active. Default: False.
        threshold (float): Threshold value for cell filtering.
        plot (bool): Whether to plot the histogram of mapped similarities. Default: True.

    Returns:
        mapped_similarity (array): shape = (n_spots/n_cells, ) depending on the mapping perspective. Cosine similarity of the shared
        genes expression profile between the deterministic cell-spot pair.
        If plot is True the histogram is drawn in a new figure (retrieve it with plt.gcf()).
    """

    # Get deterministic cell-spot pair, always (spot_idx, cell_idx)
    if flavour == "spot_to_cell":
        # Each spot assigned to a cell
        pairs_df = get_spot_cell_pair(adata_map, filter=filter, threshold=threshold)
    elif flavour == "cell_to_spot":
        # Each cell assigned to a spot
        pairs_df = get_cell_spot_pair(adata_map, filter=filter, threshold=threshold)
    else:
        raise ValueError("Invalid flavour. Choose either 'spot_to_cell' or 'cell_to_spot'.")

    # Get shared genes
    shared_genes = adata_map.uns['input_genes']

    # Convert indices/names to a form AnnData accepts
    cell_idx = pairs_df['cell index']
    spot_idx = pairs_df['spot index']

    # If they are not integer dtype, convert to list (AnnData accepts obs_names as list)
    if cell_idx.dtype.kind not in {'i', 'u'}:
        cell_idx = cell_idx.tolist()
    if spot_idx.dtype.kind not in {'i', 'u'}:
        spot_idx = spot_idx.tolist()

    # Extract expression profiles using AnnData indexing (safe with names or integers)
    # Genes are matched case-insensitively by position, without renaming the input data
    sc_profile = adata_sc[cell_idx, _gene_positions(adata_sc, shared_genes)].X
    st_profile = adata_st[spot_idx, _gene_positions(adata_st, shared_genes)].X

    # Turn to array if sparse
    if hasattr(sc_profile, "toarray"):
        sc_profile = sc_profile.toarray()
    if hasattr(st_profile, "toarray"):
        st_profile = st_profile.toarray()

    # Ensure 2D arrays
    sc_profile = np.atleast_2d(np.asarray(sc_profile, dtype=float))
    st_profile = np.atleast_2d(np.asarray(st_profile, dtype=float))

    # Compute cosine similarity (no torch)
    # Numerator: row-wise dot product as element-wise multiplication and sum over rows
    num = np.sum(sc_profile * st_profile, axis=1)  # shape (n_filtered_cells/n_spots,)
    # Denominator: product of row-wise L2 norms
    denom = np.linalg.norm(sc_profile, axis=1) * np.linalg.norm(st_profile, axis=1)
    # Cossim
    mapped_similarity = num / denom

    if plot:
        # Histogram of cosine similarities
        plt.figure()
        plt.hist(mapped_similarity, bins=np.linspace(-1, 1, 50), edgecolor="black")
        plt.title("Histogram of pair similarities")
        plt.xlabel("Cosine similarity")
        plt.ylabel("Number of pairs")

    return mapped_similarity


def _gene_positions(adata, genes):
    """
    Column positions of `genes` (lower case names) in adata, matching var names case-insensitively.
    Leaves adata untouched; if several var names differ only by case the first one is used.
    """
    positions = pd.Series(np.arange(adata.n_vars), index=adata.var_names.str.lower())
    positions = positions[~positions.index.duplicated()].reindex(genes)
    if positions.isna().any():
        raise KeyError(f"Genes not found in data: {list(positions.index[positions.isna()])}")
    return positions.to_numpy(dtype=int)
