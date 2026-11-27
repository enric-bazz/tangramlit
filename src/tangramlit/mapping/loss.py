import torch
import torch.nn as nn
from torch.nn.functional import cosine_similarity

# Gene expression similarity terms: loss term name -> lambda that weights it.
# These are the terms that depend on the genes, so they can also be scored on held-out genes.
SIMILARITY_TERMS = {
    "main_loss": "lambda_g1",
    "vg_reg": "lambda_g2",
    "sparsity_term": "lambda_sparsity_g1",
    "neighborhood_term": "lambda_neighborhood_g1",
    "getis_ord_term": "lambda_getis_ord",
    "moran_term": "lambda_moran",
    "geary_term": "lambda_geary",
}


class TangramLoss:
    """
    Stateless class to compute the Tangram loss and its components.
    It does not handle tensor device management, which is expected to be handled by the LightningModule. 
    It also does not store any state related to the training process (e.g., epoch values), which are expected to be stored in the LightningModule.
    """
    def __init__(self, hparams):
        # Inherit the hyperparameters and any other necessary components (e.g., density criterion)
        self.hparams = hparams

         # Define density criterion
        self._density_criterion = nn.KLDivLoss(reduction="sum")

    def similarity_terms(self,
                         G, G_pred,
                         voxel_weights=None,
                         getis_ord_G_star_ref=None, moran_I_ref=None, gearys_C_ref=None,
                         getis_ord_G_star_pred=None, moran_I_pred=None, gearys_C_pred=None,):
        """
        Compute the unweighted gene expression similarity terms (see SIMILARITY_TERMS).
        Each term is a (weighted) mean cosine similarity between G and G_pred, so it lies in [-1, 1]
        whatever its lambda. Only the terms whose lambda is > 0 are computed.

        Args:
            Same as __call__(), restricted to the tensors these terms need.

        Returns:
            sims: dictionary {term name: unweighted term} with the active terms only.
        """
        sims = {}

        # gene-voxel and voxel-gene terms
        if self.hparams.lambda_g1 > 0:
            sims["main_loss"] = cosine_similarity(G_pred, G, dim=0).mean()
        if self.hparams.lambda_g2 > 0:
            sims["vg_reg"] = cosine_similarity(G_pred, G, dim=1).mean()

        # sparsity-weighted term
        if self.hparams.lambda_sparsity_g1 > 0:
            mask = G != 0
            gene_sparsity = mask.sum(axis=0) / G.shape[0]
            gene_sparsity = 1 - gene_sparsity.reshape((-1,))
            sims["sparsity_term"] = ((cosine_similarity(G_pred, G, dim=0) * (1 - gene_sparsity))
                                     / (1 - gene_sparsity).sum()).sum()

        # neighbor-smoothing term
        if self.hparams.lambda_neighborhood_g1 > 0:
            sims["neighborhood_term"] = cosine_similarity(voxel_weights @ G_pred, voxel_weights @ G, dim=0).mean()

        # LISA terms
        if self.hparams.lambda_getis_ord > 0:
            sims["getis_ord_term"] = cosine_similarity(getis_ord_G_star_ref, getis_ord_G_star_pred, dim=0).mean()
        if self.hparams.lambda_moran > 0:
            sims["moran_term"] = cosine_similarity(moran_I_ref, moran_I_pred, dim=0).mean()
        if self.hparams.lambda_geary > 0:
            sims["geary_term"] = cosine_similarity(gearys_C_ref, gearys_C_pred, dim=0).mean()

        return sims

    def __call__(self, 
                 G, G_pred, 
                 M, M_probs, M_probs_filtered, 
                 F_probs, 
                 d_prior=None,
                 voxel_weights=None, 
                 ct_encode=None, neighborhood_filter=None,
                 getis_ord_G_star_ref=None, moran_I_ref=None, gearys_C_ref=None,
                 getis_ord_G_star_pred=None, moran_I_pred=None, gearys_C_pred=None,):
        """
        Compute the total loss and a dictionary of components.
        Computationally expensive terms are computed only if their respective lambda_ parameter is > 0.
        
        Args:
            G: spatial data
            G_pred: predicted spatial data
            M: alignment matrix (before softmax)
            M_probs: softmax of the alignment matrix M
            M_probs_filtered: filtered version of M_probs (if filter is used)
            F_probs: filter probabilities (if filter is used)
            d_prior: prior density (if density term is used)
            voxel_weights: voxel weights for neighborhood smoothing (if used)
            ct_encode: cell-type encoding matrix (if cell-type island term is used)
            neighborhood_filter: neighborhood filter matrix (if cell-type island term is used)  
            getis_ord_G_star_ref: reference Getis-Ord G* statistics (if used)
            moran_I_ref: reference Moran's I statistics (if used)
            gearys_C_ref: reference Geary's C statistics (if used)
            getis_ord_G_star_pred: predicted Getis-Ord G* statistics (if used)
            moran_I_pred: predicted Moran's I statistics (if used)
            gearys_C_pred: predicted Geary's C statistics (if used)
        
        Returns:    
            loss_dict: dictionary containing total loss and components
        """

        # gene expression similarity terms, each weighted by its lambda (0.0 when inactive)
        sims = self.similarity_terms(
            G=G, G_pred=G_pred, voxel_weights=voxel_weights,
            getis_ord_G_star_ref=getis_ord_G_star_ref, moran_I_ref=moran_I_ref, gearys_C_ref=gearys_C_ref,
            getis_ord_G_star_pred=getis_ord_G_star_pred, moran_I_pred=moran_I_pred, gearys_C_pred=gearys_C_pred,
        )
        weighted = {term: self.hparams[lambda_name] * sims[term] if term in sims else 0.0
                    for term, lambda_name in SIMILARITY_TERMS.items()}
        gv_term, vg_term = weighted["main_loss"], weighted["vg_reg"]
        gv_sparsity_term, gv_neighborhood_term = weighted["sparsity_term"], weighted["neighborhood_term"]
        getis_ord_term, moran_term, gearys_term = weighted["getis_ord_term"], weighted["moran_term"], weighted["geary_term"]

        # density term
        if self.hparams.filter:
            d_pred = torch.log(M_probs_filtered.sum(axis=0) / (F_probs.sum()))
        else:
            d_pred = torch.log(M_probs.sum(axis=0) / M.shape[0])
        density_term = self.hparams.lambda_d * self._density_criterion(d_pred, d_prior)

        # entropy regularizer
        regularizer_term = self.hparams.lambda_r * (torch.log(M_probs) * M_probs).sum()

        # l-norm regularization
        l1_term = self.hparams.lambda_l1 * M.abs().sum()
        l2_term = self.hparams.lambda_l2 * (M ** 2).sum()

        # cell-type island term
        if self.hparams.lambda_ct_islands > 0:
            if self.hparams.filter:
                ct_map = (M_probs_filtered.T @ ct_encode)
            else:
                ct_map = (M_probs.T @ ct_encode)
            # Create zero tensor on same device as ct_map
            zero_tensor = torch.tensor([0], dtype=torch.float32, device=ct_map.device)
            ct_island_term = self.hparams.lambda_ct_islands * (torch.max((ct_map) - (neighborhood_filter @ ct_map),
                                                                    zero_tensor).mean())
        else:
            ct_island_term = 0.0

        total_loss = (
            - gv_term
            - vg_term
            + density_term
            - regularizer_term
            + l1_term
            + l2_term
            - gv_sparsity_term
            - gv_neighborhood_term
            - getis_ord_term
            - moran_term
            - gearys_term
            + ct_island_term
        )

        # filter terms
        if self.hparams.filter:
            count_term = self.hparams.lambda_count * torch.abs(F_probs.sum() - self.hparams.target_count)

            f_reg = self.hparams.lambda_f_reg * (F_probs - F_probs * F_probs).sum()

            total_loss += count_term + f_reg 


        loss_dict = {
            "loss": total_loss,
            "main_loss": gv_term if self.hparams.lambda_g1 > 0 else None,
            "vg_reg": vg_term if self.hparams.lambda_g2 > 0 else None,
            "kl_reg": density_term if self.hparams.lambda_d > 0 else None,
            "entropy_reg": regularizer_term if self.hparams.lambda_r > 0 else None,
            "l1_term": l1_term if self.hparams.lambda_l1 > 0 else None,
            "l2_term": l2_term if self.hparams.lambda_l2 > 0 else None,
            "sparsity_term": gv_sparsity_term if self.hparams.lambda_sparsity_g1 > 0 else None,
            "neighborhood_term": gv_neighborhood_term if self.hparams.lambda_neighborhood_g1 > 0 else None,
            "getis_ord_term": getis_ord_term if self.hparams.lambda_getis_ord > 0 else None,
            "moran_term": moran_term if self.hparams.lambda_moran > 0 else None,
            "geary_term": gearys_term if self.hparams.lambda_geary > 0 else None,
            "ct_island_term": ct_island_term if self.hparams.lambda_ct_islands > 0 else None,
        }
        if self.hparams.filter:
            filter_terms = {
                "count_reg": count_term if self.hparams.lambda_count > 0 else None,
                "filt_reg": f_reg if self.hparams.lambda_f_reg > 0 else None,
            }
            loss_dict.update(filter_terms)

        return loss_dict
