#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd

# 1. Load the manual title list
print("Loading manual title list...")
manual_titles_path = '../data/processed/filtered_titles_manual.csv'

# Read the file - each title is on its own line (not comma-separated)
with open(manual_titles_path, 'r') as f:
    manual_titles = [line.strip() for line in f.readlines()]

# Get unique titles
unique_manual_titles = set(manual_titles)
print(f"Loaded {len(manual_titles)} titles, {len(unique_manual_titles)} unique titles from manual list")

# 2. Load the jobs from clusters 0 and 1
print("\nLoading jobs from clusters 0 and 1...")
cluster_jobs_path = '../data/processed/cluster_0_1_jobs.csv'
cluster_jobs_df = pd.read_csv(cluster_jobs_path)
print(f"Loaded {len(cluster_jobs_df)} jobs from clusters 0 and 1")

# 3. Filter out jobs with titles in the manual list
filtered_df = cluster_jobs_df[~cluster_jobs_df['title'].isin(unique_manual_titles)]
print(f"After filtering: {len(filtered_df)} jobs remaining")
print(f"Removed {len(cluster_jobs_df) - len(filtered_df)} jobs with titles in the manual list")

# 4. Save the filtered dataset
output_path = '../data/processed/filtered_jobs_final.csv'
filtered_df.to_csv(output_path, index=False)
print(f"Saved filtered dataset to {output_path}")

# 5. Display statistics and samples
print("\nStatistics:")
print(f"Original dataset: {len(cluster_jobs_df)} jobs")
print(f"Filtered dataset: {len(filtered_df)} jobs")
print(f"Removed: {len(cluster_jobs_df) - len(filtered_df)} jobs")

# Display some sample titles that were removed
removed_titles = set(cluster_jobs_df['title']) - set(filtered_df['title'])
print(f"\nSample of removed titles ({len(removed_titles)} total):")
sample_removed = list(removed_titles)[:20]  # Take first 20 for display
for i, title in enumerate(sample_removed, 1):
    print(f"{i}. {title}")

print("\nDone! Review the final filtered dataset.")