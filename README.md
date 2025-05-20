# CS 171 Final Project 


### Part 1:
Clustering used to create cleaned dataset of only postings related to CS/DS from [Kaggle LinkedIn dataset](https://www.kaggle.com/datasets/arshkon/linkedin-job-postings) 

### Part 2: 
Web Application that uses KNN to find job postings that most closely match a user provider resume.
Also contains a configuration page that allows custom model specification including setting the distance metric, number
of components in dimension reduction and more. 
Results include graph of top skills used in matching and their importance as well as a word cloud of key terms used. 

![Screenshot 2025-05-19 at 6 40 14 PM](https://github.com/user-attachments/assets/6286ae03-9bb0-4f97-bb12-993b7b41d0d3)
![Screenshot 2025-05-19 at 6 40 29 PM](https://github.com/user-attachments/assets/a8cec608-595c-47fa-9b0d-2cbfffa0f8f6)
![Screenshot 2025-05-19 at 6 42 19 PM](https://github.com/user-attachments/assets/7e721407-5bac-432b-a2d7-00aa5dbbcbb6)
![Screenshot 2025-05-19 at 6 39 20 PM](https://github.com/user-attachments/assets/fd4d126d-4ed4-4b62-aab7-2b8e50b59d64)

### File Structure 

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

**Note** Since data files related to this project exceed git hub upload size a compressed directory containing
all related files can be downloaded from google drive [here](https://drive.google.com/file/d/14X-rVfdDuDZ8r_yXKuAaSsecuPHL9u5S/view?usp=share_link) . This contains data files used by both the clustering and web app components. 
