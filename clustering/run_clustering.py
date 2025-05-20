"""
Run job clustering with hyperparameter tuning and visualization.
Example usage:
python run_clustering.py --tune_all --n_top_keywords 200
"""

#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Run job clustering with hyperparameter tuning and visualization.
Example usage:
python run_clustering.py --tune_all --n_top_keywords 200
"""

import argparse
import os
import sys
from pathlib import Path
from job_clustering import JobClustering

# Get the project root directory - go up from the current file's directory to the project root
# This assumes the file is in clustering/ directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..'))

# Add the project root to Python path to ensure imports work correctly
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)  # Also add the current directory

from job_clustering import JobClustering

def main():
    parser = argparse.ArgumentParser(description='Run job clustering with hyperparameter tuning')

    # Data paths - using absolute paths
    parser.add_argument('--data_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data/processed'),
                        help='Directory containing data files')
    parser.add_argument('--output_dir', type=str, default=os.path.join(PROJECT_ROOT, 'data/models'),
                        help='Directory to save model outputs')
    # ... rest of the function remains the same
    parser.add_argument('--keywords_file', type=str, default='technical_keywords_from_postings.csv',
                        help='Technical keywords CSV filename')
    parser.add_argument('--jobs_file', type=str, default='csds_filtered_clean.csv',
                        help='Job postings CSV filename')

    # Feature extraction
    parser.add_argument('--n_top_keywords', type=int, default=100,
                        help='Number of top keywords to use (try values from 100-500)')

    # Dimensionality reduction
    parser.add_argument('--dim_reduction', type=str, default='pca', choices=['pca', 'svd'],
                        help='Dimensionality reduction method')
    parser.add_argument('--n_components', type=int, default=75,
                        help='Number of components after reduction (try values from 20-100)')

    # Tuning flags
    parser.add_argument('--tune_kmeans', action='store_true', help='Tune KMeans clustering')
    parser.add_argument('--tune_hierarchical', action='store_true', help='Tune hierarchical clustering')
    parser.add_argument('--tune_dbscan', action='store_true', help='Tune DBSCAN clustering')
    parser.add_argument('--tune_all', action='store_true', help='Tune all clustering methods')

    # Clustering method to apply
    parser.add_argument('--clustering_method', type=str, default='kmeans',
                        choices=['kmeans', 'hierarchical', 'dbscan'],
                        help='Clustering method to apply after tuning')

    # Specific parameters for each method
    parser.add_argument('--n_clusters', type=int, default=8,
                        help='Number of clusters for KMeans or hierarchical')
    parser.add_argument('--linkage', type=str, default='ward',
                        choices=['ward', 'complete', 'average', 'single'],
                        help='Linkage method for hierarchical')
    parser.add_argument('--eps', type=float, default=2.0,
                        help='Epsilon parameter for DBSCAN')
    parser.add_argument('--min_samples', type=int, default=5,
                        help='Min samples parameter for DBSCAN')

    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Setup paths
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Create job clustering object
    print(f"Initializing job clustering with {args.n_top_keywords} top keywords...")
    job_clustering = JobClustering(
        data_dir=data_dir,
        output_dir=output_dir,
        keywords_file=args.keywords_file,
        jobs_file=args.jobs_file,
        n_top_keywords=args.n_top_keywords
    )

    # Create job vectors
    print("Creating job vectors...")
    job_clustering.create_job_vectors()

    # Reduce dimensions
    print(f"Reducing dimensions to {args.n_components} components using {args.dim_reduction}...")
    job_clustering.reduce_dimensions(n_components=args.n_components, method=args.dim_reduction)

    # Tune clustering parameters
    if args.tune_all or args.tune_kmeans:
        print("Tuning KMeans clustering...")
        job_clustering.tune_kmeans(job_clustering.X_reduced)

    if args.tune_all or args.tune_hierarchical:
        print("Tuning hierarchical clustering...")
        job_clustering.tune_hierarchical(job_clustering.X_reduced)

    if args.tune_all or args.tune_dbscan:
        print("Tuning DBSCAN clustering...")
        job_clustering.tune_dbscan(job_clustering.X_reduced)

    # Apply the selected clustering method with best parameters
    print(f"Applying {args.clustering_method} clustering...")

    # Use the best parameters from tuning if available
    if args.clustering_method == 'kmeans' and hasattr(job_clustering, 'best_kmeans_params'):
        args.n_clusters = job_clustering.best_kmeans_params['n_clusters']
        print(f"Using best parameters from KMeans tuning: n_clusters={args.n_clusters}")
        job_clustering.apply_best_clustering(method='kmeans', n_clusters=args.n_clusters)

    elif args.clustering_method == 'hierarchical' and hasattr(job_clustering, 'best_hierarchical_params'):
        args.n_clusters = job_clustering.best_hierarchical_params['n_clusters']
        args.linkage = job_clustering.best_hierarchical_params['linkage']
        print(f"Using best parameters from hierarchical tuning: n_clusters={args.n_clusters}, linkage={args.linkage}")
        job_clustering.apply_best_clustering(method='hierarchical', n_clusters=args.n_clusters, linkage=args.linkage)

    elif args.clustering_method == 'dbscan' and hasattr(job_clustering, 'best_dbscan_params'):
        args.eps = job_clustering.best_dbscan_params['eps']
        args.min_samples = job_clustering.best_dbscan_params['min_samples']
        print(f"Using best parameters from DBSCAN tuning: eps={args.eps}, min_samples={args.min_samples}")
        job_clustering.apply_best_clustering(method='dbscan', eps=args.eps, min_samples=args.min_samples)

    else:
        # Fall back to command line parameters if no tuning results
        if args.clustering_method == 'kmeans':
            job_clustering.apply_best_clustering(method='kmeans', n_clusters=args.n_clusters)
        elif args.clustering_method == 'hierarchical':
            job_clustering.apply_best_clustering(method='hierarchical', n_clusters=args.n_clusters,
                                                 linkage=args.linkage)
        elif args.clustering_method == 'dbscan':
            job_clustering.apply_best_clustering(method='dbscan', eps=args.eps, min_samples=args.min_samples)

    # Analyze and visualize clusters
    print("Analyzing clusters...")
    job_clustering.analyze_clusters()

    print("Visualizing clusters...")
    job_clustering.visualize_clusters()

    print("\nJob clustering completed successfully!")
    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()