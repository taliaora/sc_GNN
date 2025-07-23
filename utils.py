import torch
import torch.nn as nn
import numpy as np
import anndata as ad
import scanpy as sc
import math
from collections import Counter
from sklearn.metrics import accuracy_score
from tqdm import tqdm


def softmax(arr):
    exp_arr = np.exp(arr - np.max(arr))
    return exp_arr / np.sum(exp_arr)


def extract_subgraph(matrix, seed_node, neighbor_sizes, selection_probs):
    selected_nodes = {seed_node}
    last_layer_nodes = {seed_node}
    total_to_select = 1

    for layer_size in neighbor_sizes:
        total_to_select *= layer_size
        neighbors = matrix[list(last_layer_nodes), :].nonzero()[1]
        if len(neighbors) == 0:
            continue
        neighbors = list(set(neighbors))
        probs = selection_probs[neighbors]
        selected = np.random.choice(neighbors, min(total_to_select, len(neighbors)), replace=False, p=softmax(probs))
        last_layer_nodes = set(selected)
        selected_nodes |= last_layer_nodes

    return sorted(selected_nodes - {seed_node})


def create_batches(RNA, ATAC, neighbors=[20], batch_size=30):
    print("Splitting data into batches...")
    cell_indices = np.random.permutation(RNA.shape[1])
    total_batches = math.ceil(len(cell_indices) / batch_size)
    batch_data, node_map = [], {}

    for i in tqdm(range(total_batches)):
        start, end = i * batch_size, (i + 1) * batch_size
        current_cells = cell_indices[start:end]
        genes, peaks = [], []

        for cell in current_cells:
            rna_vals = RNA[:, cell].todense()
            rna_vals[rna_vals < 5] = 0
            gene_ids = extract_subgraph(RNA.T, cell, neighbors, np.log1p(rna_vals).flatten())
            peak_ids = extract_subgraph(ATAC.T, cell, neighbors, np.log1p(ATAC[:, cell].todense()).flatten())
            genes.extend(gene_ids)
            peaks.extend(peak_ids)
            node_map[cell] = {'genes': gene_ids, 'peaks': peak_ids}

        batch_data.append({
            'gene_index': list(set(genes)),
            'peak_index': list(set(peaks)),
            'cell_index': list(current_cells)
        })

    return batch_data, cell_indices, node_map


class LabelSmoothingLoss(nn.Module):
    def __init__(self, smoothing=0.0):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, predictions, targets):
        log_probs = torch.log_softmax(predictions, dim=-1)
        nll_loss = -log_probs.gather(dim=-1, index=targets.unsqueeze(1)).squeeze(1)
        smooth_loss = -log_probs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


def run_initial_clustering(RNA, resolution=None, n_neighbors=None, n_pcs=40, custom_embedding=None):
    print("Initializing clustering...")
    print("Recommended resolution settings based on cell count:")
    print("<=500: 0.2, 500-5000: 0.5, >5000: 0.8")

    def suggest_params(num_cells):
        if num_cells <= 500:
            return 0.2, 5
        elif num_cells <= 5000:
            return 0.5, 10
        else:
            return 0.8, 15

    adata = ad.AnnData(RNA.T, dtype='int32')
    if resolution is None or n_neighbors is None:
        resolution, n_neighbors = suggest_params(adata.n_obs)

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)

    if custom_embedding is not None:
        adata.obsm['custom'] = custom_embedding
        sc.pp.neighbors(adata, use_rep='custom', n_neighbors=n_neighbors)
    else:
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

    sc.tl.leiden(adata, resolution)
    return adata.obs['leiden']


def compute_purity(true_labels, predicted_labels):
    aligned_labels = np.zeros_like(true_labels)
    unique_true = np.unique(true_labels)
    mapping = {label: i for i, label in enumerate(unique_true)}
    true_labels = np.vectorize(mapping.get)(true_labels)

    for cluster in np.unique(predicted_labels):
        indices = predicted_labels == cluster
        majority_label = np.bincount(true_labels[indices]).argmax()
        aligned_labels[indices] = majority_label

    return accuracy_score(true_labels, aligned_labels), true_labels


def calculate_entropy(pred_labels, true_labels):
    entropy_total = 0
    total_count = len(true_labels)

    for k in set(pred_labels):
        cluster_indices = np.where(pred_labels == k)[0]
        cluster_size = len(cluster_indices)
        cluster_entropy = 0

        for j in set(true_labels):
            joint_count = np.sum(true_labels[cluster_indices] == j)
            prob = joint_count / cluster_size if cluster_size else 0
            if prob > 0:
                cluster_entropy += prob * np.log(prob)

        entropy_total += cluster_entropy * (cluster_size / total_count)

    return -entropy_total
