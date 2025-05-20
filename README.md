# CS 171 Final Project 

# Part 1:
Clustering used to create cleaned dataset of only postings related to CS/DS from [Kaggle LinkedIn dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings)

# Part 2: 
Web Application that uses KNN to find job postings that most closely match a user provider resume.
Also contains a configuration page that allows custom model specification including setting the distance metric, number
of components in dimension reduction and more. 
Results include graph of top skills used in matching and their importance as well as a word cloud of key terms used. 

```
├── JobSkillsRecommender/        # Application to parse resume and retrieve job matches using KNN
├── clustering/                  # Job clustering scripts for data preparation
│   ├── visualizations/          # Visualizations from clustering analysis
│   ├── clean_cluster_list.py    # Script to clean clustered job lists
│   ├── count.py                 # Utility for counting data properties
│   ├── data_exploration.py      # Data exploration and analysis
│   ├── extract_clustered_titles.py  # Extract job titles from clusters
│   ├── filter_job_titles.py     # Filter job titles based on criteria
│   ├── job_clustering.py        # Main clustering implementation
│   ├── job_keyword_extractor.py # Extract keywords from job postings
│   ├── run_clustering.py        # Script to run clustering process
│   └── unique.py                # Utility for finding unique entities
```
