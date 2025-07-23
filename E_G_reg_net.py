import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as sp
import anndata as ad
import pandas as pd
from collections import Counter

def build_sparse_peak_matrix(peak_tensor, gene_tensor, device):
    # Get non-zero indices and values
    row_idx, col_idx = torch.where(peak_tensor != 0)
    data = peak_tensor[row_idx, col_idx]

    row_idx = row_idx.to(device)
    col_idx = col_idx.to(device)
    data = data.to(device)

    num_genes = gene_tensor.shape[0]
    total_nonzeros = row_idx.shape[0]
    expanded_size = total_nonzeros * num_genes

    combined_rows = torch.zeros(expanded_size, dtype=torch.int32)
    combined_cols = torch.zeros(expanded_size, dtype=torch.int32)
    combined_data = torch.zeros(expanded_size)

    offset = 0
    for peak_id in range(peak_tensor.shape[0]):
        indices = (row_idx == peak_id)
        count = indices.sum().item()
        if count == 0:
            continue

        repeated_indices = torch.arange(num_genes).unsqueeze(1) + peak_id * num_genes
        repeated_indices = repeated_indices.expand(-1, count).reshape(-1)

        target_rows = (row_idx[indices] + repeated_indices - peak_id).reshape(-1)
        target_cols = col_idx[indices].repeat(num_genes)
        target_data = data[indices].repeat(num_genes)

        slice_end = offset + count * num_genes
        combined_rows[offset:slice_end] = target_rows
        combined_cols[offset:slice_end] = target_cols
        combined_data[offset:slice_end] = target_data

        offset = slice_end

    shape = (peak_tensor.shape[0] * num_genes, peak_tensor.shape[1])
    indices = torch.vstack((combined_rows, combined_cols)).long()
    return torch.sparse.FloatTensor(indices, combined_data, torch.Size(shape)).to(device)


def build_sparse_gene_matrix(peak_tensor, gene_tensor, device):
    # Get non-zero indices and values
    row_idx, col_idx = torch.where(gene_tensor != 0)
    data = gene_tensor[row_idx, col_idx]

    row_idx = row_idx.to(device)
    col_idx = col_idx.to(device)
    data = data.to(device)

    num_peaks = peak_tensor.shape[0]
    total = num_peaks * len(row_idx)

    gene_indices = (row_idx + torch.arange(num_peaks).unsqueeze(1) * gene_tensor.shape[0]).reshape(-1)
    repeated_cols = col_idx.repeat(num_peaks)
    repeated_data = data.repeat(num_peaks)

    indices = torch.vstack((gene_indices, repeated_cols))
    shape = (gene_tensor.shape[0] * num_peaks, gene_tensor.shape[1])
    return torch.sparse.FloatTensor(indices, repeated_data, torch.Size(shape)).to(device)


def compute_gene_peak_interaction(labels, node_indices, rna_mat, atac_mat, relation_mat, gene_names):
    adata = ad.AnnData(rna_mat.T, dtype='int32')
    adata.var_names = gene_names[0]
    adata.obs['label'] = pd.Categorical(np.array(labels, dtype='int64'))

    binary_matrix = sp.coo_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), list(adata.obs['label'])))
    )

    atac_selected = atac_mat[:, node_indices]
    rna_selected = rna_mat[:, node_indices]

    atac_cluster = atac_selected @ binary_matrix
    rna_cluster = rna_selected @ binary_matrix

    atac_cluster = atac_cluster / np.clip(np.sum(binary_matrix, axis=0), 1e-10, None)
    rna_cluster = rna_cluster / np.clip(np.sum(binary_matrix, axis=0), 1e-10, None)

    gene_exp = atac_cluster[np.arange(atac_cluster.shape[0]).repeat(rna_cluster.shape[0])]
    peak_exp = rna_cluster[np.tile(np.arange(rna_cluster.shape[0]), atac_cluster.shape[0])]

    gene_peak_relation = relation_mat.reshape(-1, 1).todense()
    combined = gene_exp * peak_exp * gene_peak_relation

    return combined


def compute_single_cluster_interaction(labels, rna_mat, atac_mat, relation_mat, gene_names, peak_names, cluster_id):
    adata = ad.AnnData(rna_mat.T, dtype='int32')
    adata.var_names = gene_names[0]
    adata.obs['label'] = np.array(labels, dtype='int64')

    binary_matrix = sp.coo_matrix(
        (np.ones(len(labels)), (np.arange(len(labels)), list(adata.obs['label'])))
    )

    atac_cluster = atac_mat @ binary_matrix
    rna_cluster = rna_mat @ binary_matrix

    atac_cluster /= np.clip(np.sum(binary_matrix, axis=0), 1e-10, None)
    rna_cluster /= np.clip(np.sum(binary_matrix, axis=0), 1e-10, None)

    gene_exp = atac_cluster[np.arange(atac_cluster.shape[0]).repeat(rna_cluster.shape[0])]
    peak_exp = rna_cluster[np.tile(np.arange(rna_cluster.shape[0]), atac_cluster.shape[0])]

    interaction_mask = relation_mat.reshape(-1, 1).todense()
    interaction_score = gene_exp * peak_exp * interaction_mask

    genes = np.array(gene_names[0])[np.tile(np.arange(rna_cluster.shape[0]), atac_mat.shape[0])]
    peaks = np.array(peak_names[0])[np.arange(atac_mat.shape[0]).repeat(rna_cluster.shape[0])]
    scores = np.array(interaction_score[:, cluster_id]).squeeze()

    df = pd.DataFrame({'gene': genes, 'peak': peaks, 'score': scores})
    df_filtered = df[df['score'] > 0].drop_duplicates(['gene', 'peak']).sort_values(by='score', ascending=False)

    return df, df_filtered


def calculate_egrn(labels, node_indices, rna_mat, atac_mat, relation_mat, gene_names, peak_names, threshold=0):
    print('Calculating EGRN, this may take several minutes...')
    egrn_df = pd.DataFrame(columns=['gene', 'peak', 'score', 'class'])

    try:
        interaction = compute_gene_peak_interaction(labels, node_indices, rna_mat, atac_mat, relation_mat, gene_names)
        num_clusters = len(Counter(labels))
        print(f'Number of clusters: {num_clusters}')

        genes = np.array(gene_names[0])[np.tile(np.arange(rna_mat.shape[0]), atac_mat.shape[0])]
        peaks = np.array(peak_names[0])[np.arange(atac_mat.shape[0]).repeat(rna_mat.shape[0])]

        for cluster_id in range(num_clusters):
            scores = np.array(interaction[:, cluster_id]).squeeze()
            df = pd.DataFrame({'gene': genes, 'peak': peaks, 'score': scores})
            df = df[df['score'] > threshold].drop_duplicates(['gene', 'peak']).sort_values(by='score', ascending=False)
            df['class'] = cluster_id
            egrn_df = pd.concat([egrn_df, df], ignore_index=True)

    except Exception:
        print('Fallback: using single-cluster scoring due to error or memory constraints')
        for cluster_id in range(len(Counter(labels))):
            try:
                df, filtered = compute_single_cluster_interaction(labels, rna_mat, atac_mat, relation_mat, gene_names, peak_names, cluster_id)
                filtered = filtered[filtered['score'] > threshold]
                filtered['class'] = cluster_id
                egrn_df = pd.concat([egrn_df, filtered], ignore_index=True)
            except:
                print('Memory issue encountered. Consider reducing dataset size.')
                continue

    return egrn_df
