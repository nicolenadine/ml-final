#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LinkedIn Job Data Analysis
--------------------------
This script analyzes job postings data from LinkedIn, performing exploratory data analysis,
skill extraction, and visualization of job market trends.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from wordcloud import WordCloud
import plotly.express as px
import spacy

# Create directories if they don't exist
os.makedirs('visualizations', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_spacy_model():
    """Load the SpaCy NLP model"""
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        print("Downloading spaCy model...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
    return nlp


def clean_text(text):
    """Clean and normalize job description text"""
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)

    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub(r'[^\w\s]', ' ', text)
    text = re.sub(r'\d+', ' ', text)

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def extract_country(location):
    """Extract country or region from location string"""
    if not isinstance(location, str):
        return 'Unknown'

    # Last part of location string is often the country
    parts = [part.strip() for part in location.split(',')]
    if len(parts) > 1:
        return parts[-1]
    return location


def has_skill(description, skill):
    """Check if a job description contains a specific skill"""
    if not isinstance(description, str):
        return 0
    description = description.lower()
    if f" {skill} " in f" {description} " or f" {skill}," in description or f" {skill}." in description:
        return 1
    return 0


# =============================================================================
# SKILL EXTRACTION FUNCTIONS
# =============================================================================

def extract_common_skills(descriptions, n=30):
    """Extract the most common skills from job descriptions using a predefined list"""
    # Create a list of common skills to look for
    common_skills = [
        'python', 'sql', 'java', 'r', 'c++', 'javascript', 'scala', 'matlab',
        'sas', 'excel', 'tableau', 'power bi', 'spark', 'hadoop', 'aws', 'azure',
        'gcp', 'cloud', 'docker', 'kubernetes', 'machine learning', 'deep learning',
        'ai', 'artificial intelligence', 'data science', 'statistics', 'mathematics',
        'analytics', 'visualization', 'dashboard', 'etl', 'database', 'big data',
        'nosql', 'mongodb', 'cassandra', 'postgresql', 'mysql', 'oracle',
        'data mining', 'nlp', 'natural language processing', 'computer vision',
        'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'pandas', 'numpy',
        'data analysis', 'reporting', 'bi', 'business intelligence', 'data modeling',
        'data warehouse', 'data lake', 'data engineering', 'data architecture'
    ]

    skill_counter = Counter()

    for desc in descriptions:
        if not isinstance(desc, str):
            continue
        desc = desc.lower()
        for skill in common_skills:
            if f" {skill} " in f" {desc} " or f" {skill}," in desc or f" {skill}." in desc:
                skill_counter[skill] += 1

    return skill_counter.most_common(n)


def extract_skills_tfidf(job_descriptions, min_df=0.01, max_df=0.8, top_n=100):
    """Extract skills from job descriptions using TF-IDF"""
    # Clean descriptions
    clean_descriptions = [desc if isinstance(desc, str) else "" for desc in job_descriptions]

    # Create TF-IDF vectorizer
    # Use 1-3 word phrases (unigrams, bigrams, trigrams)
    tfidf = TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=(1, 3),
        stop_words='english'
    )

    # Fit and transform job descriptions
    tfidf_matrix = tfidf.fit_transform(clean_descriptions)

    # Get feature names (potential skills)
    feature_names = tfidf.get_feature_names_out()

    # Calculate the sum of TF-IDF values for each term across all documents
    # This gives us a measure of importance
    tfidf_sum = np.array(tfidf_matrix.sum(axis=0)).flatten()

    # Sort terms by importance
    indices = np.argsort(tfidf_sum)[::-1][:top_n]
    top_terms = [(feature_names[i], tfidf_sum[i]) for i in indices]

    # Filter terms to likely be skills (you can customize this)
    # Look for technical terms, avoid common words
    skills_pattern = re.compile(r'^[a-z0-9]+([\s\-\.][a-z0-9]+)*$')
    filtered_skills = [
        (term, score) for term, score in top_terms
        if skills_pattern.match(term) and len(term) > 2
    ]

    # Count occurrences in original descriptions for better interpretability
    skill_counter = Counter()
    for term, _ in filtered_skills:
        pattern = re.compile(fr'\b{re.escape(term)}\b', re.IGNORECASE)
        for desc in clean_descriptions:
            if pattern.search(desc):
                skill_counter[term] += 1

    # Convert to DataFrame
    skills_df = pd.DataFrame(
        [(skill, count) for skill, count in skill_counter.most_common(top_n)],
        columns=['Skill', 'Count']
    )

    return skills_df


# =============================================================================
# JOB ROLE CLASSIFICATION FUNCTIONS
# =============================================================================

def get_job_types(titles):
    """Extract job types (Data Scientist, Data Analyst, etc.) from job titles"""
    job_types = []

    data_roles = [
        'data scientist', 'data analyst', 'data engineer', 'machine learning', 'ml engineer',
        'business analyst', 'business intelligence', 'bi developer', 'data architect',
        'research scientist', 'ai researcher', 'analytics', 'statistician'
    ]

    for title in titles:
        if not isinstance(title, str):
            job_types.append('Other')
            continue

        title = title.lower()
        matched = False

        for role in data_roles:
            if role in title:
                job_types.append(role)
                matched = True
                break

        if not matched:
            job_types.append('Other')

    return job_types


def extract_job_roles_clustering(job_titles, min_cluster_size=5):
    """Extract job roles by clustering similar titles"""
    # Clean and normalize job titles
    clean_titles = []
    for title in job_titles:
        if isinstance(title, str):
            # Convert to lowercase
            title = title.lower()
            # Remove special characters
            title = re.sub(r'[^\w\s]', ' ', title)
            # Remove extra whitespace
            title = re.sub(r'\s+', ' ', title).strip()
            clean_titles.append(title)
        else:
            clean_titles.append("")

    # Create n-gram vectorizer (looking for common phrases in titles)
    ngram_vectorizer = CountVectorizer(
        ngram_range=(1, 3),  # Use 1-3 word phrases
        min_df=min_cluster_size,  # Minimum document frequency
        stop_words=['at', 'in', 'for', 'and', 'or', 'the', 'of', 'to', 'with', 'a', 'an']
    )

    # Fit and transform the titles
    X = ngram_vectorizer.fit_transform(clean_titles)

    # Get feature names (n-grams)
    feature_names = ngram_vectorizer.get_feature_names_out()

    # Calculate the count of each n-gram
    ngram_counts = np.array(X.sum(axis=0)).flatten()

    # Create a dictionary of n-gram counts
    ngram_dict = {feature_names[i]: ngram_counts[i] for i in range(len(feature_names))}

    # Filter to likely job roles (longer phrases more likely to be roles)
    # Prioritize 2-3 word phrases that are more specific
    role_patterns = [
        r'data scientist',
        r'data analyst',
        r'data engineer',
        r'machine learning engineer',
        r'ml engineer',
        r'business analyst',
        r'business intelligence',
        r'data architect',
        r'research scientist',
        r'ai researcher',
        r'software engineer',
        r'full stack',
        r'frontend developer',
        r'backend developer'
    ]

    # Extract roles from titles
    roles = []
    for title in clean_titles:
        matched = False
        for pattern in role_patterns:
            if re.search(pattern, title):
                roles.append(pattern)
                matched = True
                break

        # If no predefined role matched, look for n-grams
        if not matched:
            title_ngrams = []
            for ngram in sorted(ngram_dict.keys(), key=len, reverse=True):
                if ngram in title and len(ngram.split()) > 1:
                    title_ngrams.append(ngram)

            if title_ngrams:
                # Choose the longest n-gram as the role
                roles.append(max(title_ngrams, key=len))
            else:
                # If no multi-word n-gram, use the most common word
                words = [w for w in title.split() if w not in ngram_vectorizer.stop_words]
                roles.append(words[0] if words else "other")

    # Count role frequencies
    role_counter = Counter(roles)

    # Create a DataFrame of roles
    roles_df = pd.DataFrame(
        [(role, count) for role, count in role_counter.most_common() if count >= min_cluster_size],
        columns=['Role', 'Count']
    )

    # Create a mapping from job title to role
    role_mapping = {title: role for title, role in zip(clean_titles, roles)}

    return roles_df, role_mapping


# =============================================================================
# DATA CLEANING FUNCTIONS
# =============================================================================

def clean_job_data(jobs_df):
    """Clean and preprocess the job data"""
    print("\nCleaning data...")

    # Print column names for debugging
    print("Original DataFrame columns:", jobs_df.columns.tolist())

    # Make a copy to avoid modifications to the original data
    jobs_clean = jobs_df.copy()

    # Check for expected columns and provide warnings
    expected_columns = ['title', 'company', 'location', 'description', 'work_type']
    missing_columns = [col for col in expected_columns if col not in jobs_clean.columns]

    if missing_columns:
        print(f"Warning: The following expected columns are missing: {missing_columns}")

        # Check if columns exist with different capitalization
        for missing_col in missing_columns:
            # Look for columns that match except for capitalization
            possible_matches = [col for col in jobs_clean.columns if col.lower() == missing_col.lower()]
            if possible_matches:
                print(f"  Found possible match for '{missing_col}': {possible_matches}")
                # If there's exactly one match, rename it
                if len(possible_matches) == 1:
                    jobs_clean.rename(columns={possible_matches[0]: missing_col}, inplace=True)
                    print(f"  Renamed '{possible_matches[0]}' to '{missing_col}'")

    # Convert all column names to lowercase for consistency
    jobs_clean.columns = jobs_clean.columns.str.lower()
    print("Standardized DataFrame columns:", jobs_clean.columns.tolist())

    # Fill missing values
    jobs_clean['title'] = jobs_clean['title'].fillna('Untitled Position')
    jobs_clean['company'] = jobs_clean['company'].fillna('Unknown Company')
    jobs_clean['location'] = jobs_clean['location'].fillna('Unknown Location')

    # Check if description column exists
    if 'description' in jobs_clean.columns:
        # Clean text in Description column
        jobs_clean['Description_Clean'] = jobs_clean['description'].apply(clean_text)
    else:
        print("Warning: 'description' column not found. Cannot create 'Description_Clean'")
        # Create an empty column to prevent errors later
        jobs_clean['Description_Clean'] = ""

    # Add a data role category based on job title
    jobs_clean['Data_Role'] = get_job_types(jobs_clean['title'])

    # Extract countries from locations
    jobs_clean['Country'] = jobs_clean['location'].apply(extract_country)

    # Check for work_type column, create if missing
    if 'work_type' not in jobs_clean.columns:
        print("Warning: 'work_type' column not found. Creating a placeholder.")
        # Use a simple heuristic to guess work type from description if available
        if 'description' in jobs_clean.columns:
            def guess_work_type(desc):
                if not isinstance(desc, str):
                    return 'Unknown'
                desc = desc.lower()
                if 'remote' in desc:
                    return 'Remote'
                elif 'hybrid' in desc:
                    return 'Hybrid'
                elif 'on-site' in desc or 'onsite' in desc or 'in office' in desc:
                    return 'On-site'
                else:
                    return 'Not specified'

            jobs_clean['work_type'] = jobs_clean['description'].apply(guess_work_type)
        else:
            jobs_clean['work_type'] = 'Not specified'

    # Check data after cleaning
    print("\nMissing values after cleaning:")
    print(jobs_clean.isnull().sum())

    return jobs_clean


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def visualize_job_titles(jobs_clean):
    """Visualize the most common job titles"""
    print("\nMost common job titles:")
    title_counts = jobs_clean['title'].value_counts().head(20)
    print(title_counts)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=title_counts.values, y=title_counts.index)
    plt.title('Top 20 Job Titles', fontsize=16)
    plt.xlabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/top_job_titles.png')
    plt.close()

    # Data role distribution
    data_role_counts = jobs_clean['Data_Role'].value_counts()
    plt.figure(figsize=(14, 8))
    ax = sns.barplot(x=data_role_counts.values, y=data_role_counts.index)
    ax.set_title('Distribution of Data Roles', fontsize=16)
    ax.set_xlabel('Count', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/data_role_distribution.png')
    plt.close()

    return data_role_counts


def visualize_companies(jobs_clean):
    """Visualize the top companies with most job postings"""
    print("\nTop companies with most job postings:")
    company_counts = jobs_clean['company'].value_counts().head(20)
    print(company_counts)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=company_counts.values, y=company_counts.index)
    plt.title('Top 20 Companies by Job Postings', fontsize=16)
    plt.xlabel('Number of Job Postings', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/top_companies.png')
    plt.close()

    return company_counts


def visualize_locations(jobs_clean):
    """Visualize the top locations for job postings"""
    print("\nTop locations:")
    location_counts = jobs_clean['Country'].value_counts().head(20)
    print(location_counts)

    plt.figure(figsize=(14, 8))
    sns.barplot(x=location_counts.values, y=location_counts.index)
    plt.title('Top 20 Countries/Regions for Data Jobs', fontsize=16)
    plt.xlabel('Number of Job Postings', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/top_locations.png')
    plt.close()

    return location_counts


def visualize_job_types(jobs_clean):
    """Visualize job type distribution (Remote/Hybrid/Onsite)"""
    print("\nJob type distribution (Remote/Hybrid/Onsite):")

    # Check if work_type column exists and has data
    if 'work_type' not in jobs_clean.columns:
        print("Warning: 'work_type' column not found in the DataFrame")
        return None

    # Count non-null values
    non_null_count = jobs_clean['work_type'].count()
    if non_null_count == 0:
        print("Warning: 'work_type' column contains no data")
        return None

    job_type_counts = jobs_clean['work_type'].value_counts()
    print(job_type_counts)

    # Only create visualization if there's data
    if len(job_type_counts) > 0:
        plt.figure(figsize=(10, 6))
        plt.pie(job_type_counts, labels=job_type_counts.index, autopct='%1.1f%%', startangle=90,
                colors=sns.color_palette("pastel"))
        plt.axis('equal')
        plt.title('Job Type Distribution (Remote/Hybrid/Onsite)', fontsize=16)
        plt.tight_layout()
        plt.savefig('visualizations/job_type_distribution.png')
        plt.close()
    else:
        print("Warning: No job type data available for pie chart")

    # Create a heatmap of data roles vs job types
    # Add safety checks to prevent error with empty data
    if non_null_count > 0 and jobs_clean['Data_Role'].count() > 0:
        # Create the crosstab
        role_type_matrix = pd.crosstab(jobs_clean['Data_Role'], jobs_clean['work_type'])

        # Check if the matrix has data
        if role_type_matrix.size > 0 and not role_type_matrix.empty:
            plt.figure(figsize=(12, 10))
            sns.heatmap(role_type_matrix, annot=True, fmt='d', cmap='YlGnBu')
            plt.title('Data Roles vs. Job Types', fontsize=16)
            plt.tight_layout()
            plt.savefig('visualizations/role_vs_type_heatmap.png')
            plt.close()
        else:
            print("Warning: Role-type matrix is empty, cannot create heatmap")
    else:
        print("Warning: Insufficient data for role vs job type heatmap")

    return job_type_counts

def visualize_skills(jobs_clean, common_skills):
    """Visualize the most common skills mentioned in job descriptions"""
    # Create a DataFrame of skills
    skills_df = pd.DataFrame(common_skills, columns=['Skill', 'Count'])
    print("\nMost common skills mentioned:")
    print(skills_df)

    # Visualize the most common skills
    plt.figure(figsize=(14, 8))
    sns.barplot(x='Count', y='Skill', data=skills_df)
    plt.title('Top 30 Skills Mentioned in Job Descriptions', fontsize=16)
    plt.xlabel('Number of Mentions', fontsize=14)
    plt.tight_layout()
    plt.savefig('visualizations/top_skills.png')
    plt.close()

    # Create word cloud of skills
    plt.figure(figsize=(16, 8))
    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=100, contour_width=3, contour_color='steelblue')
    wordcloud.generate_from_frequencies({skill: count for skill, count in common_skills})
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title('Word Cloud of Skills', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/skills_wordcloud.png')
    plt.close()

    return skills_df


def visualize_skill_distribution_by_role(jobs_clean, common_skills):
    """Visualize skill distribution by data role"""
    # Select top 10 skills for analysis
    top_skills = [skill for skill, _ in common_skills[:10]]

    # Create columns for each top skill
    for skill in top_skills:
        jobs_clean[f'has_{skill}'] = jobs_clean['Description_Clean'].apply(lambda x: has_skill(x, skill))

    # Create a DataFrame showing skill distribution by data role
    skill_by_role = pd.DataFrame()
    for role in jobs_clean['Data_Role'].unique():
        role_df = jobs_clean[jobs_clean['Data_Role'] == role]
        skill_percentages = []
        for skill in top_skills:
            percentage = role_df[f'has_{skill}'].sum() / len(role_df) * 100
            skill_percentages.append(percentage)
        skill_by_role[role] = skill_percentages

    skill_by_role.index = top_skills

    # Visualize skills by data role as a heatmap
    plt.figure(figsize=(14, 10))
    sns.heatmap(skill_by_role, annot=True, fmt='.1f', cmap='YlGnBu')
    plt.title('Percentage of Job Postings Mentioning Top Skills by Data Role', fontsize=16)
    plt.tight_layout()
    plt.savefig('visualizations/skills_by_role_heatmap.png')
    plt.close()

    return skill_by_role, top_skills


def perform_text_analysis(jobs_clean):
    """Perform TF-IDF analysis and dimensionality reduction on job descriptions"""
    print("\nPerforming text analysis with TF-IDF...")

    # Subset the data for better visualization (random sample)
    sample_size = min(1000, len(jobs_clean))
    jobs_sample = jobs_clean.sample(sample_size, random_state=42)

    # TF-IDF Vectorization
    tfidf_vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words='english',
        ngram_range=(1, 2)
    )

    # Fit and transform the job descriptions
    tfidf_matrix = tfidf_vectorizer.fit_transform(jobs_sample['Description_Clean'])
    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

    # Apply dimensionality reduction
    svd = TruncatedSVD(n_components=2, random_state=42)
    job_points = svd.fit_transform(tfidf_matrix)

    # Create a DataFrame with the reduced dimensions
    job_viz_df = pd.DataFrame({
        'x': job_points[:, 0],
        'y': job_points[:, 1],
        'Data_Role': jobs_sample['Data_Role'].values,
        'Job_Type': jobs_sample['work_type'].values,
        'Title': jobs_sample['title'].values,
        'Company': jobs_sample['company'].values
    })

    # Visualize job descriptions in 2D space by data role
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x='x', y='y', hue='Data_Role', data=job_viz_df, alpha=0.7)
    plt.title('Job Descriptions in 2D Space by Data Role', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/job_descriptions_by_role.png')
    plt.close()

    # Visualize job descriptions in 2D space by job type
    plt.figure(figsize=(14, 10))
    sns.scatterplot(x='x', y='y', hue='Job_Type', data=job_viz_df, alpha=0.7)
    plt.title('Job Descriptions in 2D Space by Job Type (Remote/Hybrid/Onsite)', fontsize=16)
    plt.xlabel('Component 1', fontsize=14)
    plt.ylabel('Component 2', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('visualizations/job_descriptions_by_type.png')
    plt.close()

    return job_viz_df


def create_interactive_visualizations(jobs_clean, job_viz_df, top_skills):
    """Create interactive visualizations with Plotly"""
    print("\nCreating interactive visualizations...")

    # Interactive scatter plot of job descriptions
    fig = px.scatter(
        job_viz_df, x='x', y='y',
        color='Data_Role',
        hover_name='Title',
        hover_data=['Company', 'Job_Type'],
        title='Interactive Map of Job Descriptions'
    )
    # For notebook display
    fig.show()

    # Interactive visualization of skills by data role
    role_skill_data = []
    for role in jobs_clean['Data_Role'].unique():
        if role == 'Other':
            continue
        role_df = jobs_clean[jobs_clean['Data_Role'] == role]
        for skill in top_skills:
            percentage = role_df[f'has_{skill}'].sum() / len(role_df) * 100
            role_skill_data.append({
                'Data_Role': role,
                'Skill': skill,
                'Percentage': percentage
            })

    skill_role_df = pd.DataFrame(role_skill_data)

    # Create a grouped bar chart
    fig = px.bar(
        skill_role_df,
        x='Skill',
        y='Percentage',
        color='Data_Role',
        title='Skill Distribution by Data Role',
        labels={'Percentage': 'Percentage of Job Postings', 'Skill': 'Skill'},
        barmode='group'
    )
    # For notebook display
    fig.show()

    return skill_role_df


def perform_geographic_analysis(jobs_clean):
    """Perform geographic analysis of job postings"""
    print("\nPerforming geographic analysis...")

    # Get top 10 countries
    top_countries = jobs_clean['Country'].value_counts().head(10).index.tolist()

    # Filter to top countries and get role distribution in each
    country_role_data = []
    for country in top_countries:
        country_df = jobs_clean[jobs_clean['Country'] == country]
        role_counts = country_df['Data_Role'].value_counts()
        for role, count in role_counts.items():
            if role == 'Other':
                continue
            country_role_data.append({
                'Country': country,
                'Data_Role': role,
                'Count': count
            })

    country_role_df = pd.DataFrame(country_role_data)

    # Create a grouped bar chart for top countries and roles
    fig = px.bar(
        country_role_df,
        x='Country',
        y='Count',
        color='Data_Role',
        title='Data Role Distribution in Top 10 Countries',
        labels={'Count': 'Number of Job Postings', 'Country': 'Country'},
        barmode='stack'
    )
    # For notebook display
    fig.show()

    return country_role_df


def save_processed_data(jobs_clean):
    """Save the cleaned and processed data"""
    print("\nSaving cleaned data...")
    jobs_clean.to_csv('data/processed/linkedin_jobs_cleaned.csv', index=False)

    # Create processed dataset for skill extraction
    skill_extraction_df = jobs_clean[['title', 'company', 'location', 'Description_Clean', 'work_type', 'Data_Role']]
    skill_extraction_df.to_csv('data/processed/skill_extraction_data.csv', index=False)

    print("\nExploratory data analysis complete. Visualizations saved to 'visualizations/' directory.")


def summarize_findings(jobs_clean, data_role_counts, company_counts, location_counts, job_type_counts, common_skills):
    """Summarize key findings from the analysis"""
    print("\nSummary of Key Findings:")
    print("------------------------")
    print(f"1. The dataset contains {jobs_clean.shape[0]} job postings from LinkedIn.")
    print(f"2. The most common job role is {data_role_counts.index[0]} with {data_role_counts.values[0]} postings.")
    print(
        f"3. The company with the most job postings is {company_counts.index[0]} with {company_counts.values[0]} listings.")
    print(f"4. The top location for data jobs is {location_counts.index[0]} with {location_counts.values[0]} postings.")
    print(
        f"5. The most common job type is {job_type_counts.index[0]} with {job_type_counts.values[0]} postings ({job_type_counts.values[0] / len(jobs_clean) * 100:.1f}%).")
    print(f"6. The most frequently mentioned skill is {common_skills[0][0]} with {common_skills[0][1]} mentions.")
    print(
        "7. There are clear skill patterns across different data roles, with certain skills being more prevalent in specific roles.")
    print("8. Dimensionality reduction reveals clustering patterns in job descriptions by role type.")

    print("\nRecommendations for Next Steps:")
    print("------------------------------")
    print("1. Implement advanced NLP techniques for more comprehensive skill extraction")
    print("2. Develop a more refined skill taxonomy based on extracted skills")
    print("3. Apply clustering algorithms to discover natural skill groupings")
    print("4. Train a job category classifier based on extracted skills")
    print("5. Build a recommendation system based on skill similarity")
    print("6. Create interactive visualizations for the web interface")


# =============================================================================
# MAIN FUNCTION
# =============================================================================
# Update the main function to be more robust
def main():

    import os
    print("Current working directory:", os.getcwd())
    """Main function to run the LinkedIn job data analysis"""
    print("LinkedIn Job Data Analysis")
    print("=========================")

    # Load spaCy model for NLP tasks
    try:
        nlp = load_spacy_model()
    except Exception as e:
        print(f"Warning: Could not load spaCy model: {str(e)}")
        print("Continuing without NLP capabilities...")

    # Load data
    print("\nLoading data...")
    try:
        jobs_df = pd.read_csv('../data/raw/linkedin_jobs.csv')
        jobs_df.info()
        print(f"Loaded {len(jobs_df)} job postings.")
        print("Columns in the dataset:", jobs_df.columns.tolist())
    except FileNotFoundError:
        print("Error: Could not find data file")
        print("Checking for other CSV files in the data directory...")
        import glob
        csv_files = glob.glob('data/**/*.csv', recursive=True)
        if csv_files:
            print(f"Found these CSV files: {csv_files}")
            print("Please specify which one to use or move the correct file to data/raw/linkedin_jobs.csv")
        else:
            print("No CSV files found in the data directory.")
        return
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return

    # Clean data with enhanced validation
    jobs_clean = clean_job_data(jobs_df)

    print("\nPerforming exploratory data analysis...")

    # More robust function calling with error handling
    try:
        data_role_counts = visualize_job_titles(jobs_clean)
        company_counts = visualize_companies(jobs_clean)
        location_counts = visualize_locations(jobs_clean)

        # This function had the error, so wrap it in try/except
        try:
            job_type_counts = visualize_job_types(jobs_clean)
        except Exception as e:
            print(f"Error in job type visualization: {str(e)}")
            job_type_counts = pd.Series() if 'work_type' not in jobs_clean else jobs_clean['work_type'].value_counts()

        # Continue with other analyses only if we have description data
        if 'description' in jobs_clean.columns and jobs_clean['Description_Clean'].astype(bool).sum() > 0:
            # Extract and visualize skills
            print("\nExtracting skills mentioned in job descriptions...")
            all_descriptions = jobs_clean['Description_Clean'].tolist()
            common_skills = extract_common_skills(all_descriptions, n=30)
            skills_df = visualize_skills(jobs_clean, common_skills)

            # Analyze skill distribution by role
            try:
                skill_by_role, top_skills = visualize_skill_distribution_by_role(jobs_clean, common_skills)

                # Advanced text analysis
                job_viz_df = perform_text_analysis(jobs_clean)

                # Interactive visualizations
                skill_role_df = create_interactive_visualizations(jobs_clean, job_viz_df, top_skills)

                # Geographic analysis
                country_role_df = perform_geographic_analysis(jobs_clean)

                # Also extract skills using TF-IDF for comparison
                print("\nExtracting skills using TF-IDF for comparison...")
                tfidf_skills_df = extract_skills_tfidf(jobs_clean['description'].tolist(), min_df=0.01, max_df=0.8,
                                                       top_n=50)
                print("\nTop skills extracted with TF-IDF:")
                print(tfidf_skills_df.head(20))

                # Visualize TF-IDF extracted skills
                plt.figure(figsize=(14, 8))
                sns.barplot(x='Count', y='Skill', data=tfidf_skills_df.head(20))
                plt.title('Top 20 Skills Extracted with TF-IDF', fontsize=16)
                plt.xlabel('Number of Job Postings', fontsize=14)
                plt.tight_layout()
                plt.savefig('visualizations/top_skills_tfidf.png')
                plt.close()
            except Exception as e:
                print(f"Error in advanced analysis: {str(e)}")
                print("Skipping advanced analyses...")
        else:
            print("Warning: No description data available. Skipping skill extraction and advanced analyses.")
            common_skills = []

        # Save processed data
        save_processed_data(jobs_clean)

        # Summarize findings
        try:
            summarize_findings(jobs_clean, data_role_counts, company_counts, location_counts,
                               job_type_counts, common_skills)
        except Exception as e:
            print(f"Error in summary generation: {str(e)}")
            print("Could not generate summary.")

        print("\nAnalysis complete!")

    except Exception as e:
        print(f"Unexpected error in analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        print("Analysis incomplete due to errors.")


if __name__ == "__main__":
    main()
