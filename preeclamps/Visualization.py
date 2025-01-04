import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from proteindata_preprocessing import ProteinDataPreprocessing
from clinicdata_preprocessing import ClinicDataPreprocessing
from data_filter import FilterData
from class_dataprocessByCluster import Clustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import umap


def plot_pca(data):
    """ PCA plot""" 
    X = data.iloc[:,:-1].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    # Get the explained variance ratio for each compfiltered.columnsonent
    explained_variance_ratio = pca.explained_variance_ratio_
    print("Explained variance ratio:", pca.explained_variance_ratio_[:10])
    plt.figure(figsize=(12, 4))
    plt.plot(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio, marker='o', linestyle='--', color='b')
    plt.title('Elbow Plot: Explained Variance Ratio vs. Number of Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)
    plt.xticks(range(1, len(explained_variance_ratio) + 1))
    plt.show()
    return X_scaled

def group_pca_plot(X_scaled, data, hue_column='group_key', n_components=2):
    """
    Perform PCA and create a scatter plot for the first two principal components.
    Parameters:
    - X_scaled (array-like): Scaled feature matrix.
    - data (pd.DataFrame): DataFrame containing metadata (e.g., group labels).
    - hue_column (str): Column name in 'data' used for coloring the plot.
    - n_components (int): Number of PCA components to compute (default: 2).

    Returns:
    - None: Displays the PCA scatter plot.
    """
    # Perform PCA
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(X_scaled)
    # Create a DataFrame for PCA results
    pca_df = pd.DataFrame(pca_result, columns=[f'PC{i+1}' for i in range(n_components)])
    pca_df[hue_column] = data[hue_column].values
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x='PC1', y='PC2', data=pca_df, hue=hue_column, palette='Set1')
    plt.title('PCA Plot: First vs. Second Component')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(title='Group', loc='best')
    plt.show()
    return pca_df

def remove_pca_outliers(pca_df, data, X_scaled, group_key='group2', threshold=3):
    """
    Identify and remove PCA outliers based on Z-scores.

    Parameters:
    - pca_df (pd.DataFrame): DataFrame containing PCA results with 'PC1', 'PC2', and 'group_key'.
    - data (pd.DataFrame): Original data containing metadata.
    - X_scaled (np.ndarray): Scaled feature matrix.
    - group_key (str): Group key to filter the PCA data for outlier detection.
    - threshold (float): Z-score threshold for identifying outliers (default: 3).

    Returns:
    - df_clean (pd.DataFrame): DataFrame with outliers removed.
    - X_clean (np.ndarray): Scaled feature matrix with outliers removed.
    - valid_outlier_indices (list): List of valid outlier indices.
    """
    group_df = pca_df[pca_df['group_key'] == group_key]
    z_scores = np.abs(stats.zscore(group_df[['PC1', 'PC2']]))
    outliers = (z_scores > threshold).any(axis=1)
    outlier_indices = group_df[outliers].index
    print("Outlier indices:", outlier_indices)
    valid_outlier_indices = list(set(outlier_indices).intersection(set(data.index)))
    print("Valid Outlier Indices:", valid_outlier_indices)
    df_clean = data.drop(index=valid_outlier_indices)
    X_clean = X_scaled[~data.index.isin(valid_outlier_indices)]
    print("Cleaned Data Shape:", df_clean.shape)
    print("Cleaned X Shape:", X_clean.shape)
    return df_clean, X_clean, valid_outlier_indices

def umap_plot(X_scaled, data, n_components=2, n_neighbors=15, min_dist=0.1, random_state=42):
    """
    Perform UMAP dimensionality reduction and create a scatter plot.

    Parameters:
    - X_scaled: Scaled feature matrix (numpy array or DataFrame).
    - data: Original data containing 'group_key' for coloring clusters.
    - n_components: Number of dimensions for UMAP projection (default=2).
    - n_neighbors: Number of neighboring points used in UMAP (default=15).
    - min_dist: Minimum distance between points in UMAP space (default=0.1).
    - random_state: Seed for reproducibility (default=42).
    """
    umap_reducer = umap.UMAP(n_components=n_components, 
                            n_neighbors=n_neighbors, 
                            min_dist=min_dist, 
                            random_state=random_state)
    umap_result = umap_reducer.fit_transform(X_scaled)
    umap_df = pd.DataFrame(umap_result, columns=[f'UMAP{i+1}' for i in range(n_components)])
    umap_df['group_key'] = data['group_key'].values
    plt.figure(figsize=(10, 7))
    sns.scatterplot(data=umap_df, x='UMAP1', y='UMAP2', hue='group_key', palette='Set1')
    plt.title('UMAP Projection of Clustered Data')
    plt.xlabel('UMAP1')
    plt.ylabel('UMAP2')
    plt.legend(title='Group', loc='best')
    plt.show()

def plot_tsne(X_scaled, data, model=None, perplexity=30, random_state=42):
    """
    Function to perform t-SNE and plot the results.
    
    Parameters:
    - X_scaled: The scaled feature matrix (e.g., after scaling your data)
    - data: The original data with ground truth labels or clustering labels
    - model: The clustering model used (optional), for fallback labels
    - perplexity: The perplexity parameter for t-SNE (default is 30)
    - random_state: Random state for reproducibility (default is 42)
    """
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
    tsne_result = tsne.fit_transform(X_scaled)
    tsne_df = pd.DataFrame(tsne_result, columns=['t-SNE1', 't-SNE2'])
    if 'group_key' in data.columns:
        tsne_df['group_key'] = data['group_key'].values
    elif model is not None:
        tsne_df['group_key'] = model.labels_  # Use model labels if available
    else:
        raise ValueError("No valid group labels found")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='t-SNE1',
        y='t-SNE2',
        data=tsne_df,
        hue='group_key',
        palette='Set2',
        s=100,
        edgecolor='black',
        alpha=0.7
    )
    plt.title('t-SNE Visualization of Clusters')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(title='Cluster', loc='best')
    plt.show()

# def plot_clusters_meanshift(self,df_clean):
#     """ Plot clusters"""
#     # Plotting the clusters
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x='PC1', y='PC2', data=df_clean, hue='cluster', palette='Set1')
#     plt.title('Clusters in PCA Space')
#     plt.xlabel('PC1')
#     plt.ylabel('PC2')
#     plt.legend(title='Cluster', loc='best')
#     plt.show()

