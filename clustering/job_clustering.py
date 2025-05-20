import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import joblib
import seaborn as sns
from matplotlib import cm
import argparse

warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

class JobClustering:
    """
    Class for clustering job postings based on technical keywords.
    Uses the TF-IDF vectors from the job_keyword_extractor.py to create job vectors.
    """
    def __init__(self, data_dir='data/processed', output_dir='data/models',
                 keywords_file='technical_keywords_from_postings.csv', jobs_file='csds_filtered_clean.csv',
                 n_top_keywords=200):
        self.data_dir = os.path.join(PROJECT_ROOT, data_dir)
        self.output_dir = os.path.join(PROJECT_ROOT, output_dir)
        self.keywords_file = keywords_file
        self.jobs_file = jobs_file
        self.n_top_keywords = n_top_keywords

        os.makedirs(self.output_dir, exist_ok=True)
        self.load_data()

    def _ensure_output_dir(self) -> Path:
        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def _evaluate_clustering(self, X, labels):
        unique_labels = set(labels)
        if len(unique_labels - {-1}) < 2 or np.sum(labels != -1) < 2:
              return 0, 0, float('inf')
        valid = labels != -1
        X_use = X[valid]
        y_use = np.array(labels)[valid]
        return (
            silhouette_score(X_use, y_use),
            calinski_harabasz_score(X_use, y_use),
            davies_bouldin_score(X_use, y_use)
        )

    def _save_plot(self, fig_or_plt, filename: str):
        output_dir = self._ensure_output_dir()
        path = output_dir / filename
        fig_or_plt.savefig(path)
        print(f"Saved plot to {path}")

    def _save_csv(self, df: pd.DataFrame, filename: str):
        output_dir = self._ensure_output_dir()
        path = output_dir / filename
        df.to_csv(path, index=False)
        print(f"Saved CSV to {path}")

    def reduce_dimensions(self, n_components=50, method='pca'):
        """
        Reduce the dimensionality of the feature vectors.

        Parameters:
        -----------
        n_components : int
            Number of components to reduce to
        method : str
            Method to use for dimensionality reduction ('pca' or 'svd')
        """
        print(f"Reducing dimensions to {n_components} using {method}...")

        if not hasattr(self, 'X'):
            raise ValueError("Feature vectors have not been created. Call create_job_vectors() first.")

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=42)
        elif method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Unknown dimensionality reduction method: {method}")

        X_reduced = reducer.fit_transform(X_scaled)

        print(f"Reduced dimensions from {self.X.shape[1]} to {X_reduced.shape[1]}")
        if hasattr(reducer, 'explained_variance_ratio_'):
            print(f"Explained variance ratio: {np.sum(reducer.explained_variance_ratio_):.4f}")

        self.X_reduced = X_reduced
        self.reducer = reducer
        self.scaler = scaler

        return X_reduced

    def load_data(self):
        print(f"Loading data from {self.data_dir}...")
        keywords_path = os.path.join(self.data_dir, self.keywords_file)
        print(f"Looking for keywords file at: {keywords_path}")
        self.keywords_df = pd.read_csv(keywords_path)
        print(f"Loaded {len(self.keywords_df)} technical keywords")
        self.top_keywords = self.keywords_df.sort_values('importance_score', ascending=False).head(self.n_top_keywords)
        print(f"Selected top {self.n_top_keywords} keywords by importance score")
        self.top_terms = self.top_keywords['term'].tolist()

        jobs_path = os.path.join(self.data_dir, self.jobs_file)
        print(f"Looking for jobs file at: {jobs_path}")
        self.jobs_df = pd.read_csv(jobs_path)
        print(f"Loaded {len(self.jobs_df)} job postings")

        required_cols = ['title', 'description', 'skills_desc']
        for col in required_cols:
            if col not in self.jobs_df.columns:
                print(f"Warning: Required column '{col}' not found in the dataset.")

        self.jobs_df['description'] = self.jobs_df['description'].fillna('')
        self.jobs_df['skills_desc'] = self.jobs_df['skills_desc'].fillna('')

    def create_job_vectors(self):
        print(f"Creating job vectors using top {self.n_top_keywords} keywords...")

        def preprocess_technical_text(text):
            if not isinstance(text, str):
                return ""
            text = text.lower()
            replacements = {
                'c\+\+': 'cplusplus', 'c#': 'csharp', '\.net': 'dotnet',
                'node\.js': 'nodejs', 'react\.js': 'reactjs',
                'type script': 'typescript', 'type-script': 'typescript',
                'java script': 'javascript', 'java-script': 'javascript',
                'objective-c': 'objectivec', 'objective c': 'objectivec',
                'machine learning': 'machinelearning', 'deep learning': 'deeplearning',
                'natural language processing': 'nlp', 'computer vision': 'computervision',
                'artificial intelligence': 'ai'
            }
            import re
            for pattern, replacement in replacements.items():
                text = re.sub(pattern, replacement, text)
            text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

        print("Preprocessing job text...")
        processed_title = self.jobs_df['title'].apply(preprocess_technical_text)
        processed_description = self.jobs_df['description'].apply(preprocess_technical_text)
        processed_skills = self.jobs_df['skills_desc'].apply(preprocess_technical_text)

        combined_text = (
            processed_title + ' ' + processed_title + ' ' +
            processed_description + ' ' +
            processed_skills + ' ' + processed_skills + ' ' + processed_skills
        )

        print("Creating feature vectors...")
        X = np.zeros((len(self.jobs_df), len(self.top_terms)))

        for i, job_text in enumerate(tqdm(combined_text)):
            for j, term in enumerate(self.top_terms):
                import re
                pattern = r'\b' + re.escape(term) + r'\b'
                count = len(re.findall(pattern, job_text))
                X[i, j] = count

        print(f"Created feature vectors with shape {X.shape}")
        self.X = X
        self.vectors_df = pd.DataFrame(X, columns=self.top_terms)
        self.vectors_df['job_id'] = self.jobs_df['job_id'].values
        return X

    def tune_kmeans(self, X_reduced, k_range=range(2, 21), n_init=10):
        silhouette_scores, calinski_scores, davies_bouldin_scores, inertia_scores = [], [], [], []

        for k in tqdm(k_range):
            kmeans = KMeans(n_clusters=k, init='k-means++', n_init=n_init, random_state=42)
            clusters = kmeans.fit_predict(X_reduced)
            sil, cal, db = self._evaluate_clustering(X_reduced, clusters)
            silhouette_scores.append(sil)
            calinski_scores.append(cal)
            davies_bouldin_scores.append(db)
            inertia_scores.append(kmeans.inertia_)

        results = pd.DataFrame({
            'k': list(k_range),
            'silhouette': silhouette_scores,
            'calinski_harabasz': calinski_scores,
            'davies_bouldin': davies_bouldin_scores,
            'inertia': inertia_scores
        })

        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes[0, 0].plot(results['k'], results['silhouette'], 'o-')
        axes[0, 0].set_title('Silhouette Score')
        axes[0, 1].plot(results['k'], results['calinski_harabasz'], 'o-')
        axes[0, 1].set_title('Calinski-Harabasz Index')
        axes[1, 0].plot(results['k'], results['davies_bouldin'], 'o-')
        axes[1, 0].set_title('Davies-Bouldin Index')
        axes[1, 1].plot(results['k'], results['inertia'], 'o-')
        axes[1, 1].set_title('Inertia (Elbow Method)')
        plt.tight_layout()

        self._save_plot(plt, 'kmeans_tuning.png')
        self._save_csv(results, 'kmeans_tuning_results.csv')

        # Find the best k based on silhouette score
        best_k = results.loc[results['silhouette'].idxmax(), 'k']

        print(f"Best parameters: n_clusters={int(best_k)}")

        # Store best parameters
        self.best_kmeans_params = {
            'n_clusters': int(best_k)
        }

        return results

    def tune_hierarchical(self, X_reduced, k_range=range(2, 21), linkages=['ward', 'complete', 'average']):
        records = []
        for linkage in linkages:
            for k in tqdm(k_range, desc=f"Linkage={linkage}"):
                model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
                clusters = model.fit_predict(X_reduced)
                sil, cal, db = self._evaluate_clustering(X_reduced, clusters)
                records.append(
                    {'linkage': linkage, 'k': k, 'silhouette': sil, 'calinski_harabasz': cal, 'davies_bouldin': db})

        results_df = pd.DataFrame.from_records(records)
        plt.figure(figsize=(15, 10))
        for linkage in linkages:
            subset = results_df[results_df['linkage'] == linkage]
            plt.plot(subset['k'], subset['silhouette'], 'o-', label=linkage)
        plt.title('Hierarchical Clustering: Silhouette Score')
        plt.xlabel('k')
        plt.ylabel('Silhouette Score')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        self._save_plot(plt, 'hierarchical_tuning.png')
        self._save_csv(results_df, 'hierarchical_tuning_results.csv')

        # Find the combination with the best silhouette score
        best_idx = results_df['silhouette'].idxmax()
        best_params = results_df.loc[best_idx, ['linkage', 'k']]

        print(f"Best parameters: linkage={best_params['linkage']}, n_clusters={best_params['k']}")

        # Store best parameters
        self.best_hierarchical_params = {
            'linkage': best_params['linkage'],
            'n_clusters': int(best_params['k'])
        }

        return results_df

    def tune_dbscan(self, X_reduced, eps_range=np.arange(0.5, 5.1, 0.5), min_samples_range=[5, 15, 25, 35, 50, 75]):
        records = []
        for eps in eps_range:
            for min_samples in min_samples_range:
                model = DBSCAN(eps=eps, min_samples=min_samples)
                clusters = model.fit_predict(X_reduced)
                sil, cal, db = self._evaluate_clustering(X_reduced, clusters)
                n_clusters = len(set(clusters)) - (1 if -1 in clusters else 0)
                n_noise = list(clusters).count(-1)
                noise_ratio = n_noise / len(clusters)
                records.append({
                    'eps': eps, 'min_samples': min_samples, 'n_clusters': n_clusters,
                    'n_noise': n_noise, 'noise_ratio': noise_ratio,
                    'silhouette': sil, 'calinski_harabasz': cal, 'davies_bouldin': db
                })

        results_df = pd.DataFrame(records)
        pivot = results_df.pivot_table(index='min_samples', columns='eps', values='silhouette')
        plt.figure(figsize=(15, 10))
        sns.heatmap(pivot, annot=True, fmt='.2f', cmap='viridis')
        plt.title('DBSCAN: Silhouette Score by Parameters')
        plt.xlabel('Epsilon')
        plt.ylabel('Min Samples')
        plt.tight_layout()

        self._save_plot(plt, 'dbscan_tuning.png')
        self._save_csv(results_df, 'dbscan_tuning_results.csv')

        # Filter out configurations with too many noise points (more than 50%)
        valid_results = results_df[results_df['noise_ratio'] < 0.5]

        if len(valid_results) > 0:
            # Find best parameters (highest silhouette score)
            best_idx = valid_results['silhouette'].idxmax()
            best_params = valid_results.loc[best_idx]

            print(f"Best parameters: eps={best_params['eps']}, min_samples={int(best_params['min_samples'])}, "
                  f"resulting in {int(best_params['n_clusters'])} clusters with "
                  f"{best_params['noise_ratio']:.1%} noise points")

            # Store best parameters
            self.best_dbscan_params = {
                'eps': best_params['eps'],
                'min_samples': int(best_params['min_samples'])
            }
        else:
            print("Warning: No valid DBSCAN configurations found with reasonable noise levels.")
            # Set some default parameters
            self.best_dbscan_params = {
                'eps': 1.0,
                'min_samples': 5
            }

        return results_df

    def apply_best_clustering(self, method='kmeans', **params):
        print(f"Applying {method} clustering with parameters: {params}")

        if not hasattr(self, 'X_reduced'):
            raise ValueError("Dimensionality reduction has not been performed. Call reduce_dimensions() first.")

        if method == 'kmeans':
            model = KMeans(n_clusters=params.get('n_clusters', 8), init='k-means++', n_init=params.get('n_init', 10),
                           random_state=42)
        elif method == 'hierarchical':
            model = AgglomerativeClustering(n_clusters=params.get('n_clusters', 7),
                                            linkage=params.get('linkage', 'ward'))
        elif method == 'dbscan':
            model = DBSCAN(eps=params.get('eps', 2.0), min_samples=params.get('min_samples', 5))
        else:
            raise ValueError(f"Unknown clustering method: {method}")

        clusters = model.fit_predict(self.X_reduced)
        self.jobs_df['cluster'] = clusters

        print("Cluster distribution:")
        total = len(self.jobs_df)
        for cluster, count in self.jobs_df['cluster'].value_counts().sort_index().items():
            print(f"  Cluster {cluster}: {count} jobs ({count / total * 100:.2f}%)")

        self._ensure_output_dir()
        joblib.dump(model, Path(self.output_dir) / f"{method}_model.joblib")
        print(f"Saved {method} model to {Path(self.output_dir) / f'{method}_model.joblib'}")
        self.jobs_df.to_csv(Path(self.output_dir) / 'jobs_with_clusters.csv', index=False)
        print(f"Saved jobs with cluster assignments to {Path(self.output_dir) / 'jobs_with_clusters.csv'}")

        self.model = model
        self.method = method
        self.clusters = clusters
        return clusters

    def analyze_clusters(self):
        print("Analyzing clusters...")

        if not hasattr(self, 'clusters'):
            raise ValueError("Clustering has not been performed. Call apply_best_clustering() first.")

        unique_clusters = sorted(set(self.clusters))
        if -1 in unique_clusters:
            unique_clusters.remove(-1)
            print("Note: Cluster -1 (noise points) will be excluded from analysis")

        records = []
        for cluster in unique_clusters:
            cluster_jobs = self.jobs_df[self.jobs_df['cluster'] == cluster]
            cluster_mean = self.X[self.jobs_df['cluster'] == cluster].mean(axis=0)
            top_idx = np.argsort(cluster_mean)[-10:][::-1]
            top_keywords = [self.top_terms[i] for i in top_idx]
            top_titles = cluster_jobs['title'].value_counts().head(5).index.tolist()
            records.append({
                'cluster': cluster,
                'size': len(cluster_jobs),
                'percentage': len(cluster_jobs) / len(self.jobs_df) * 100,
                'top_keywords': top_keywords,
                'top_titles': top_titles
            })

        df = pd.DataFrame(records)
        self._save_csv(df, 'cluster_analysis.csv')
        print("\nCluster Analysis Summary:")
        for _, row in df.iterrows():
            print(f"\nCluster {row['cluster']} ({int(row['size'])} jobs, {row['percentage']:.2f}%):")
            print(f"  Top Keywords: {', '.join(row['top_keywords'])}")
            print(f"  Top Titles: {', '.join(row['top_titles'])}")

        self.cluster_analysis = df
        return df

    def visualize_clusters(self):
        print("Visualizing clusters...")

        if not hasattr(self, 'clusters'):
            raise ValueError("Clustering has not been performed. Call apply_best_clustering() first.")

        pca = PCA(n_components=2, random_state=42)
        X_2d = pca.fit_transform(self.X_reduced)

        df = pd.DataFrame({
            'x': X_2d[:, 0],
            'y': X_2d[:, 1],
            'cluster': self.clusters,
            'title': self.jobs_df['title'].values
        })

        plt.figure(figsize=(12, 10))
        color_map = cm.get_cmap('rainbow', len(set(self.clusters)))
        for idx, cluster in enumerate(sorted(set(self.clusters))):
            data = df[df['cluster'] == cluster]
            color = 'black' if cluster == -1 else color_map(idx)
            marker = 'x' if cluster == -1 else 'o'
            plt.scatter(data['x'], data['y'], c=[color], marker=marker, alpha=0.7,
                        label=f'Cluster {cluster}' if cluster != -1 else 'Noise')

        plt.title(f'Job Clusters ({self.method})')
        plt.xlabel(f'PCA Component 1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        plt.ylabel(f'PCA Component 2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self._save_plot(plt, 'cluster_visualization.png')
        self._save_csv(df, 'cluster_visualization_data.csv')
        return df

def main():
    parser = argparse.ArgumentParser(description='Job Clustering')
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data/processed'),
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data/models'),
                        help='Directory to save model outputs')
    parser.add_argument('--keywords_file', type=str, default='technical_keywords_from_postings.csv',
                        help='Technical keywords CSV')
    parser.add_argument('--jobs_file', type=str, default='csds_filtered_clean.csv', help='Job postings CSV')
    parser.add_argument('--n_top_keywords', type=int, default=100, help='Number of top keywords to use')
    parser.add_argument('--dim_reduction', type=str, default='pca', choices=['pca', 'svd'],
                        help='Dimensionality reduction method')
    parser.add_argument('--n_components', type=int, default=50,
                        help='Number of components for dimensionality reduction')
    parser.add_argument('--clustering_method', type=str, default='kmeans',
                        choices=['kmeans', 'hierarchical', 'dbscan'],
                        help='Clustering method')
    parser.add_argument('--tune', action='store_true', help='Tune clustering parameters')
    parser.add_argument('--n_clusters', type=int, default=7, help='Number of clusters (for KMeans or Hierarchical)')
    parser.add_argument('--linkage', type=str, default='ward', choices=['ward', 'complete', 'average'],
                        help='Linkage method (for Hierarchical)')
    parser.add_argument('--eps', type=float, default=2.0, help='Epsilon parameter (for DBSCAN)')
    parser.add_argument('--min_samples', type=int, default=5, help='Min samples parameter (for DBSCAN)')
    args = parser.parse_args()

    job_clustering = JobClustering(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        keywords_file=args.keywords_file,
        jobs_file=args.jobs_file,
        n_top_keywords=args.n_top_keywords
    )

    job_clustering.create_job_vectors()
    job_clustering.reduce_dimensions(n_components=args.n_components, method=args.dim_reduction)

    if args.tune:
        if args.clustering_method == 'kmeans':
            job_clustering.tune_kmeans(job_clustering.X_reduced)
        elif args.clustering_method == 'hierarchical':
            job_clustering.tune_hierarchical(job_clustering.X_reduced)
        elif args.clustering_method == 'dbscan':
            job_clustering.tune_dbscan(job_clustering.X_reduced)

    if args.clustering_method == 'kmeans':
        job_clustering.apply_best_clustering(method='kmeans', n_clusters=args.n_clusters)
    elif args.clustering_method == 'hierarchical':
        job_clustering.apply_best_clustering(method='hierarchical', n_clusters=args.n_clusters,
                                             linkage=args.linkage)
    elif args.clustering_method == 'dbscan':
        job_clustering.apply_best_clustering(method='dbscan', eps=args.eps, min_samples=args.min_samples)

    job_clustering.analyze_clusters()
    job_clustering.visualize_clusters()
    print("Job clustering completed successfully!")


if __name__ == "__main__":
    main()

