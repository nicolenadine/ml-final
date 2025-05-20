JobSkillRecommender
A machine learning system that matches resumes to job postings using KNN (K-Nearest Neighbors) algorithm and natural language processing techniques.
Project Overview
JobSkillRecommender is an advanced job matching system designed to help CS/DS job-seekers find the most relevant job opportunities based on their resumes. The system uses machine learning techniques, specifically K-Nearest Neighbors (KNN), to compare resume content with job descriptions and find the best matches.
The system extracts key technical terms from both resumes and job postings, creates feature vectors based on these terms, and uses KNN to find the most similar jobs for a given resume. It also provides detailed explanations of why certain jobs were matched, highlighting the specific skills and terms that contributed to the match.
Key Features

Resume Parsing: Extract text from PDF resumes with fallback OCR support for image-based PDFs
Term-Based Feature Engineering: Uses a comprehensive dictionary of technical terms organized by category
Machine Learning Matching: KNN algorithm with options for different distance metrics
Dimensionality Reduction: Option to use PCA or SVD for better performance
Match Explanation: Provides detailed information about why matches were made
Web Interface: Flask-based web application for easy job matching
Configurable Parameters: User interface for tweaking matching algorithm parameters

Project Structure
JobSkillRecommender/
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
├── data/                        # Data directory
│   ├── models/                  # Saved ML models
│   ├── processed/               # Processed datasets
│   ├── raw/                     # Raw input data
│   └── vectors/                 # Feature vectors for jobs
├── tests/                       # Test scripts
│   ├── test_outputs/            # Test output files
│   ├── test_integration.py      # Integration tests
│   ├── test_parser.py           # Tests for resume parser
│   └── test_term_matching.py    # Tests for term matching
├── uploads/                     # Directory for uploaded resumes
├── app.py                       # Flask web application
├── custom_terms.py              # Custom terms for job-resume matching
├── job_vectorizer.py            # Processes job postings into term vectors
├── knn_matcher.py               # KNN machine learning for matching
├── requirements.txt             # Project dependencies
├── resume_analyzer.py           # Analyzes resumes for structured info
├── resume_job_system.py         # Main system integrating all components
├── resume_match.py              # CLI script for matching resumes to jobs
├── resume_parser.py             # Parser for extracting text from resumes
└── term_utils.py                # Utilities for term vector creation

Core Components
1. Resume Processing

resume_parser.py: Extracts text from PDF resumes using multiple strategies for maximum reliability
resume_analyzer.py: Analyzes resume text to extract structured information and create feature vectors

2. Job Processing

job_vectorizer.py: Processes job postings into term vectors for efficient similarity matching
job_keyword_extractor.py: Extracts technical keywords specific to CS/DS roles from job postings

3. Matching Engine

knn_matcher.py: Implements KNN algorithm for matching resumes to jobs with options for different metrics
term_utils.py: Utility functions for creating term vectors and calculating similarity
custom_terms.py: Comprehensive dictionary of technical terms organized by category

4. Web Interface

app.py: Flask web application with upload form and results visualization
config.html: Configuration page for the matching algorithm parameters
index.html: Home page with upload form
results.html: Results page showing job matches with visualizations

Usage
Web Interface

Run the Flask application:
python app.py

Open a web browser and navigate to http://localhost:5000
Upload a resume (PDF format)
Review the job matches displayed on the results page

Command Line
Match a resume to jobs:
python resume_match.py path/to/resume.pdf --metric cosine --neighbors 10
Options:

--metric: Distance metric to use (cosine, euclidean, manhattan)
--neighbors: Number of neighbors for KNN
--dim-reduction: Use dimensionality reduction
--top: Number of top matches to return
--output: Output file path for results
--verbose: Print verbose output

Algorithm Details
The matching system works through the following process:

Text Extraction: The resume is parsed using PyMuPDF and pdfplumber with OCR fallback
Feature Engineering: Both resume and job descriptions are converted to term vectors based on the presence of technical terms
Dimensionality Reduction: (Optional) PCA or SVD reduces the vector space dimensions
KNN Matching: The K-Nearest Neighbors algorithm finds the most similar job postings
Result Explanation: The system explains which terms contributed most to each match

The KNN configuration includes:

Distance metrics: cosine, euclidean, manhattan, etc.
Algorithm options: auto, ball_tree, kd_tree, brute
Dimensionality reduction methods: PCA, SVD
Feature standardization options

Future Improvements

Enhanced Resume Parsing: Improve text extraction from complex PDF formats
More Advanced NLP: Implement word embeddings or transformers for better semantic matching
Skill Gap Analysis: Identify missing skills in resumes compared to desired jobs
Interactive Visualizations: Add more interactive visualizations of job matches
User Feedback Loop: Incorporate user feedback on match quality to improve the algorithm
Multi-language Support: Extend support for resumes and job descriptions in multiple languages
Job Recommendation API: Create a REST API for integrating with other applications
Mobile App: Develop a mobile application for on-the-go job matching

File Descriptions
Main System Components

app.py: Flask web application for the user interface
resume_job_system.py: Main system integrating all components
knn_matcher.py: KNN implementation for matching resumes to jobs
resume_parser.py: Parser for extracting text from resume PDFs
job_vectorizer.py: Creates and manages job term vectors
custom_terms.py: Dictionary of technical terms by category
term_utils.py: Utilities for term vector creation and similarity calculation
resume_match.py: Command-line script for matching resumes to jobs

Data Processing Components

job_clustering.py: Implements clustering for job categorization
job_keyword_extractor.py: Extracts relevant keywords from job postings
resume_analyzer.py: Extracts structured data from resumes
filter_job_titles.py: Filters job titles for relevant positions
clean_cluster_list.py: Cleans clustered job lists for better matching
data_exploration.py: Scripts for exploring and analyzing job data

Web Interface

index.html: Home page with resume upload form
config.html: Configuration page for ML parameters
results.html: Results page showing matched jobs