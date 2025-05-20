#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# 1. Load cluster analysis data
print("Loading cluster analysis file...")
cluster_df = pd.read_csv('../data/models/pca_n100/cluster_visualization_data.csv')
print(f"Loaded cluster data with {len(cluster_df)} rows and columns: {cluster_df.columns.tolist()}")

# 2. Extract titles from clusters 0 and 1
target_clusters = [0, 1]
filtered_clusters = cluster_df[cluster_df['cluster'].isin(target_clusters)]
print(f"Found {len(filtered_clusters)} jobs in clusters {target_clusters}")

# 3. Get unique job titles
unique_titles = set(filtered_clusters['title'])
print(f"Found {len(unique_titles)} unique job titles")

# 4. Load original dataset
print("Loading original dataset...")
original_df = pd.read_csv('../data/processed/csds_filtered.csv')
print(f"Loaded original dataset with {len(original_df)} rows")

# 5. Filter to get only jobs with titles from clusters 0 and 1
filtered_df = original_df[original_df['title'].isin(unique_titles)]
print(f"Created filtered dataset with {len(filtered_df)} rows")

# 6. Save results
filtered_df.to_csv('../data/processed/cluster_0_1_jobs.csv', index=False)
print("Saved filtered dataset to ../data/processed/cluster_0_1_jobs.csv")

with open('../data/processed/cluster_0_1_titles.txt', 'w') as f:
    for title in sorted(unique_titles):
        f.write(f"{title}\n")
print("Saved list of unique titles to ../data/processed/cluster_0_1_titles.txt")

# 7. Display sample of titles for quick review
print("\nSample of job titles from clusters 0 and 1:")
for i, title in enumerate(sorted(list(unique_titles))[:20], 1):
    print(f"{i}. {title}")

print("\nDone! Review the job titles to confirm they're the ones you want to remove.")