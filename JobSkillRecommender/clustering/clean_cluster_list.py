#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd
import os
import re

print("Starting job filtering process...")

# 1. Load the manually filtered titles (titles to keep)
print("Loading manually filtered titles...")
manual_file_path ='../data/processed/unique_titles_final.csv'
try:
    # Since it's not comma separated, read as plain text file
    with open(manual_file_path, 'r') as f:
        titles_to_keep = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(titles_to_keep)} titles to keep")
except FileNotFoundError:
    print(f"Error: File not found at {manual_file_path}")
    exit(1)

# 2. Load cluster_0_1_jobs.csv
print("Loading cluster_0_1_jobs.csv...")
cluster_jobs_path = '../data/processed/cluster_0_1_jobs.csv'
try:
    cluster_df = pd.read_csv(cluster_jobs_path)
    original_count = len(cluster_df)
    print(f"Loaded {original_count} jobs from clusters 0 and 1")
except FileNotFoundError:
    print(f"Error: File not found at {cluster_jobs_path}")
    exit(1)

# 3. Remove rows where title matches any in titles_to_keep
print("Filtering cluster jobs...")
# Keep rows where title is NOT in titles_to_keep
filtered_cluster_df = cluster_df[~cluster_df['title'].isin(titles_to_keep)]
removed_count = original_count - len(filtered_cluster_df)
print(f"Removed {removed_count} rows from cluster jobs")
print(f"Original count: {original_count}, New count: {len(filtered_cluster_df)}")

# 4. Extract all titles from filtered dataset to to_remove.txt
print("Extracting titles to remove...")
titles_to_remove = sorted(filtered_cluster_df['title'].unique())
print(f"Found {len(titles_to_remove)} unique titles to remove from clusters")

# 5. Add additional filtering for specific keywords
keywords_to_filter = ["Retail", "Sales", "Manufacturing", "Electrical", "Process", "Mechanical"]
print(f"Adding additional filtering for keywords: {', '.join(keywords_to_filter)}")

# 6. Load csds_filtered.csv to find additional titles containing keywords
print("\nLoading csds_filtered.csv...")
csds_path = '../data/processed/csds_filtered.csv'
try:
    csds_df = pd.read_csv(csds_path)
    original_count = len(csds_df)
    print(f"Loaded {original_count} jobs from csds_filtered.csv")
except FileNotFoundError:
    print(f"Error: File not found at {csds_path}")
    exit(1)

# Create regex pattern for case-insensitive matching
pattern = '|'.join(keywords_to_filter)
regex = re.compile(pattern, re.IGNORECASE)

# Find all titles containing the keywords
keyword_titles = csds_df[csds_df['title'].str.contains(regex, regex=True, na=False)]['title'].unique()
print(f"Found {len(keyword_titles)} unique titles containing filtered keywords")

# 7. Combine both sets of titles to remove
combined_titles_to_remove = sorted(set(titles_to_remove) | set(keyword_titles))
print(f"Combined: {len(combined_titles_to_remove)} total unique titles to remove")

# 8. Save titles to to_remove.txt
to_remove_path = 'data/processed/to_remove.txt'
print(f"Saving {len(combined_titles_to_remove)} titles to {to_remove_path}...")
with open(to_remove_path, 'w') as f:
    for title in combined_titles_to_remove:
        f.write(f"{title}\n")
print(f"Successfully saved titles to {to_remove_path}")

# 9. Filter csds_filtered.csv to remove jobs with titles in combined_titles_to_remove
print("Filtering csds_filtered.csv...")
filtered_csds_df = csds_df[~csds_df['title'].isin(combined_titles_to_remove)]
removed_count = original_count - len(filtered_csds_df)
print(f"Removed {removed_count} rows from csds_filtered.csv")
print(f"Original count: {original_count}, New count: {len(filtered_csds_df)}")

# 10. Save filtered csds_filtered.csv
output_path = '../data/processed/csds_filtered_clean.csv'
print(f"Saving filtered dataset to {output_path}...")
filtered_csds_df.to_csv(output_path, index=False)
print(f"Successfully saved filtered dataset to {output_path}")

# 11. Show some stats
print("\nSummary:")
print(f"- Kept {len(titles_to_keep)} manually selected titles")
print(f"- Identified {len(titles_to_remove)} titles to remove from clusters")
print(f"- Identified {len(keyword_titles)} titles containing filtered keywords")
print(f"- Combined total of {len(combined_titles_to_remove)} titles to remove")
print(f"- Removed {removed_count} jobs from the main dataset")
print(f"- Final dataset has {len(filtered_csds_df)} jobs")

# 12. Show sample of removed titles
print("\nSample of removed titles:")
sample_size = min(20, len(combined_titles_to_remove))
for i, title in enumerate(sorted(combined_titles_to_remove)[:sample_size], 1):
    print(f"{i}. {title}")

print("\nDone!")