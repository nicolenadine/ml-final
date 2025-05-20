## This is a quick overview of the files used in the clustering portion of this project

#Scripts

### Data Cleaning and Filtering

- filter_job_titles.py: Filters raw job postings to identify CS/DS roles, removing non-technical positions. 
   Creates csds_filtered.csv with filtered results.
- clean_cluster_list.py: Removes job titles specified in a manually filtered list, 
    plus those containing keywords like "Retail", "Sales", etc. Creates csds_filtered_clean.csv.
- count.py: Counts and filters jobs from clusters based on manually identified titles. 
    Provides statistics about removed titles and saves the filtered dataset.
- extract_clustered_titles.py: Extracts job titles from specific clusters (0 and 1), 
    saves them to cluster_0_1_titles.txt, and creates a filtered dataset of jobs from these clusters.
- unique.py: Extracts unique job titles from a filtered list and saves them to unique_titles_final.csv.

### Data Analysis and Feature Engineering

data_exploration.py: Comprehensive script for exploring job data including:
  - Cleaning and preprocessing job listings
  - Extracting skills and keywords from descriptions
  - Analyzing job titles, companies, locations, and work types
  - Visualizing trends and distributions
  - Creating interactive visualizations and reports

job_keyword_extractor.py: Extracts technical terms from job descriptions using TF-IDF and 
   specialized processing to identify relevant keywords in CS/DS positions.
