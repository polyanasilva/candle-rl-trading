import itertools
import pandas as pd
from sklearn.cluster import KMeans

class ClusteringModel:
    def __init__(self, n_body_clusters=7, n_lower_shadow_clusters=5, n_upper_shadow_clusters=5):
        self.models = {
            'body': KMeans(n_clusters=n_body_clusters, random_state=42),
            'lower_shadow': KMeans(n_clusters=n_lower_shadow_clusters, random_state=42),
            'upper_shadow': KMeans(n_clusters=n_upper_shadow_clusters, random_state=42)
        }
        self.n_body_clusters = n_body_clusters
        self.n_lower_shadow_clusters = n_lower_shadow_clusters
        self.n_upper_shadow_clusters = n_upper_shadow_clusters

    def fit_predict(self, data):
        data['body_cluster'] = self.models['body'].fit_predict(data[['body']])
        data['lower_shadow_cluster'] = self.models['lower_shadow'].fit_predict(data[['lower_shadow']])
        data['upper_shadow_cluster'] = self.models['upper_shadow'].fit_predict(data[['upper_shadow']])
        return data

    def reorder_clusters(self, data, feature, cluster_col):
        cluster_means = data.groupby(cluster_col)[feature].mean().sort_values()
        mapping = {old: new for new, old in enumerate(cluster_means.index)}
        data[cluster_col] = data[cluster_col].map(mapping)
        return data

    def assign_combined_clusters(self, data):
        all_combinations = list(itertools.product(
            range(self.n_body_clusters),
            range(self.n_lower_shadow_clusters),
            range(self.n_upper_shadow_clusters)
        ))

        unique_combinations = pd.DataFrame(all_combinations, columns=['body_cluster', 'lower_shadow_cluster', 'upper_shadow_cluster'])
        unique_combinations['combined_cluster'] = range(1, len(unique_combinations) + 1)

        combination_mapping = {
            (row['body_cluster'], row['lower_shadow_cluster'], row['upper_shadow_cluster']): row['combined_cluster']
            for _, row in unique_combinations.iterrows()
        }

        data['cluster'] = data.apply(
            lambda row: combination_mapping[
                (row['body_cluster'], row['lower_shadow_cluster'], row['upper_shadow_cluster'])
            ],
            axis=1
        )
        return data

# === Uso ===
# clustering = ClusteringModel()
# data = clustering.fit_predict(data)
# data = clustering.reorder_clusters(data, 'body', 'body_cluster')
# data = clustering.reorder_clusters(data, 'lower_shadow', 'lower_shadow_cluster')
# data = clustering.reorder_clusters(data, 'upper_shadow', 'upper_shadow_cluster')
# data = clustering.assign_combined_clusters(data)