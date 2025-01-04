import numpy as np
import pandas as pd
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import matplotlib.pyplot as plt

class DataProcessing_by_Scanpy():
    def __init__(self, data):
        self.data = data

    def read_data(self, df):
        X = df.drop('group_key', axis=1).values 
        # Create the AnnData object
        adata = sc.AnnData(X)       
        adata.obs['group_key'] = df['group_key']
        return adata
    
    def preprocess(self, adata):
        sc.pp.normalize_total(adata)
        # Logarithmize the data
        sc.pp.log1p(adata)
        # Perform PCA for dimensionality reduction
        sc.tl.pca(adata)  
        # Compute neighborhood graph
        sc.pp.neighbors(adata)
        return adata

    def clustering(self, adata):
        # Use Leiden clustering
        sc.tl.leiden(adata, resolution=1.25) 
        # Perform t-SNE
        sc.tl.tsne(adata)  
        # Perform UMAP
        sc.tl.umap(adata)  
        # Plot UMAP
        self.plot_umap(adata)
    
    def plot_umap(self, adata):
        """Plot the UMAP with group annotations."""
        plt.figure(figsize=(8, 6))
        sc.pl.umap(adata, color='leiden', title="UMAP Plot by Group", show=True)
