from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN, OPTICS, MeanShift, Birch, AffinityPropagation, BisectingKMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from proteindata_preprocessing import ProteinDataPreprocessing
from clinicdata_preprocessing import ClinicDataPreprocessing
from data_filter import FilterData
import pandas as pd


class Clustering:
    def __init__(self, data, cluster_type='kmeans', n_clusters=3, n_init=10, max_iter=300):
        self.data = data
        self.cluster_type = cluster_type.lower()
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter

    def scale_and_reduce(self,n_components=3):
        scaler = StandardScaler()
        pca = PCA(n_components=n_components)
        self.data = pca.fit_transform(scaler.fit_transform(self.data))

    def fit_cluster(self):
        clustering_algorithms = {
            'kmeans': KMeans(n_clusters=self.n_clusters, n_init=self.n_init, max_iter=self.max_iter),
            'agglomerative': AgglomerativeClustering(n_clusters=self.n_clusters),
            'meanshift': MeanShift(),
            'birch': Birch(n_clusters=self.n_clusters),
            'affinity': AffinityPropagation(),
            'bisectingkmeans': BisectingKMeans(n_clusters=self.n_clusters)
        }

        if self.cluster_type not in clustering_algorithms:
            raise ValueError(f"Unknown clustering type: {self.cluster_type}")

        model = clustering_algorithms[self.cluster_type]
        model.fit(self.data)
        self.model = model
        return model

    def evaluate(self):
        if hasattr(self.model, 'labels_') and len(set(self.model.labels_)) > 1:
            score = silhouette_score(self.data, self.model.labels_)
            print(f"the {self.cluster_type} clustering method")
            print(f"Silhouette Score: {score:.2f}")
        else:
            print("Silhouette score not applicable for this clustering method.")


if __name__ == "__main__":
    # Data Preprocessing
    clinic_file = '/students/2024-2025/master/pre_eclampsy/pre-eclampsia.xlsx'
    protein_file = '/students/2024-2025/master/pre_eclampsy/20241112_MS_pre-eclampsia.xlsx'
    data_processor = ProteinDataPreprocessing(protein_file)
    df = data_processor.load_protein_data()
    df = data_processor.delete_unrelevant_features(df)
    df = data_processor.rename_rows(df)
    data_processor.check_features_distribution(df)
    df = data_processor.impute_missing_values(df)
    unique_df = data_processor.unique_samples(df)

    # Clinic Data Processing
    clinic_processor = ClinicDataPreprocessing()
    group_data = {}
    for i, group in enumerate(clinic_processor.load_data(clinic_file), start=1):
        if 'Patiëntcode' in group.columns:
            group['Patiëntcode'].replace('NaN', pd.NA, inplace=True)
            group.dropna(subset=['Patiëntcode'], inplace=True)
            group_data[f'group{i}'] = group['Patiëntcode'].astype(str).str.upper().tolist()
    uniqueid_df = data_processor.add_tag(unique_df, group_data)

    # Filtering
    filtered_data = FilterData(uniqueid_df).filter_by_groups_std()
    print(filtered_data)
    data = filtered_data.iloc[:,:-1]
    # Clustering
    for i in ['kmeans', 'agglomerative', 'meanshift', 'birch', 'affinity', 'bisectingkmeans']:
        clustering = Clustering(data, cluster_type=i, n_clusters=6)
        clustering.scale_and_reduce()
        model = clustering.fit_cluster()
        clustering.evaluate()
        
