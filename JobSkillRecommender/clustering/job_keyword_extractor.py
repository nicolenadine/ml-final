import pandas as pd
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)


def extract_technical_keywords(df, title_col='title', desc_col='description', skills_col='skills_desc'):
    """
    Extract technical keywords specific to CS/DS roles from job postings.
    Uses a targeted approach to find technical terms, enhanced with a predefined list.
    """
    print(f"Processing {len(df)} job postings with enhanced technical term detection")

    # Preprocess text - preserve technical terms better
    def preprocess_technical_text(text):
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Replace common technical patterns before removing special chars
        # This preserves terms like c++, c#, .net, etc.
        replacements = {
            'c\\+\\+': 'cplusplus',
            'c#': 'csharp',
            '\\.net': 'dotnet',
            'node\\.js': 'nodejs',
            'react\\.js': 'reactjs',
            'type script': 'typescript',
            'type-script': 'typescript',
            'java script': 'javascript',
            'java-script': 'javascript',
            'objective-c': 'objectivec',
            'objective c': 'objectivec',
            'machine learning': 'machinelearning',
            'deep learning': 'deeplearning',
            'natural language processing': 'nlp',
            'computer vision': 'computervision',
            'artificial intelligence': 'ai'
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Remove special characters (keeping alphanumerics and spaces)
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    # Apply preprocessing
    print("Preprocessing text...")
    processed_title = df[title_col].apply(preprocess_technical_text)
    processed_description = df[desc_col].apply(preprocess_technical_text)
    processed_skills = df[skills_col].apply(preprocess_technical_text)

    # Create combined text - giving more weight to title and skills
    combined_text = (
            processed_title + ' ' + processed_title + ' ' +
            processed_description + ' ' +
            processed_skills + ' ' + processed_skills + ' ' + processed_skills
    )

    # Predefined list of CS/DS technical terms - languages, libraries, tools, frameworks
    specific_tech_terms = [
        "python", "java", "javascript", "cplusplus", "csharp", "r", "sql", "typescript",
        "scala", "go", "kotlin", "swift", "bash", "matlab", "ruby", "php", "perl",
        "julia", "rust", "dart", "haskell", "objectivec", "shell", "groovy", "assembly",
        "pandas", "numpy", "scikit", "tensorflow", "pytorch", "keras", "xgboost", "lightgbm",
        "matplotlib", "seaborn", "plotly", "statsmodels", "opencv", "nltk", "spacy",
        "flask", "django", "fastapi", "streamlit", "shiny", "react", "angular", "vue",
        "nextjs", "nodejs", "git", "github", "gitlab", "docker", "kubernetes", "jenkins",
        "airflow", "aws", "azure", "gcp", "databricks", "snowflake", "tableau", "powerbi",
        "jupyter", "vscode", "intellij", "postman", "figma", "notion", "jira", "trello",
        "slack", "mlflow", "huggingface", "mysql", "postgresql", "sqlite", "oracle", "mongodb",
        "cassandra", "redis", "dynamodb", "elasticsearch", "bigquery", "hive", "presto",
        "dremio", "sqlalchemy", "dbt", "spark", "graphql", "nosql", "olap", "oltp",
        "supervised", "unsupervised", "reinforcement", "neural", "cnn", "rnn", "transformers",
        "generative", "embeddings", "hyperparameter", "validation", "precision", "recall",
        "roc", "f1", "overfitting", "underfitting", "ensemble", "probability", "bayesian",
        "hypothesis", "algebra", "calculus", "timeseries", "pca", "regression", "statistics",
        "anova", "markov", "optimization", "kafka", "hadoop", "etl", "datawarehouse", "datalake",
        "batch", "streaming", "beam", "luigi", "agile", "scrum", "oauth", "jwt", "https", "tls",
        "ssh", "penetration", "cicd", "terraform", "serverless", "lambda", "microservices",
        "rest", "api", "pytest", "junit", "testng", "mocking", "blockchain", "iot", "quantum",
        "federated", "gdpr", ".NET", "iOS"
    ]

    # Core technical domain terms - broader technical concepts
    core_tech_terms = [
        'data', 'software', 'design', 'engineer', 'engineering', 'development',
        'systems', 'technical', 'cloud', 'network', 'developer', 'test', 'product',
        'system', 'application', 'analysis', 'testing', 'technology', 'automation',
        'database', 'tools', 'security', 'infrastructure', 'code', 'programming',
        'algorithm', 'computation', 'computing', 'interface', 'platform',
        'architecture', 'frontend', 'backend', 'fullstack', 'devops', 'swe',
        'deployment', 'web', 'mobile', 'desktop', 'saas'
    ]

    # Combine all technical terms
    all_tech_terms = specific_tech_terms + core_tech_terms

    # Create custom stopwords list that preserves technical terms
    standard_stopwords = set(stopwords.words('english'))

    # Extended technical terms to preserve - basic tech abbreviations
    tech_abbreviations = [
        'c', 'r', 'go', 'net', 'sql', 'api', 'aws', 'gcp', 'ui', 'ux', 'ml',
        'ai', 'bi', 'qa', 'db', 'ide', 'php', 'css', 'html', 'js', 'ts'
    ]

    # Terms to always exclude (non-technical terms that shouldn't be in results)
    exclusion_terms = [
        # Generic job terms
        'job', 'work', 'team', 'company', 'position', 'role', 'experience',
        'skills', 'knowledge', 'ability', 'responsibilities', 'requirements',
        'qualifications', 'opportunity', 'candidate', 'project', 'business',
        'service', 'year', 'years', 'strong', 'required', 'preferred', 'using',
        'use', 'used', 'provide', 'working', 'environment', 'time', 'must',
        'including', 'new', 'help', 'develop', 'lead', 'support', 'manage',

        # Non-CS terms that appeared in previous results
        'benefits', 'equipment', 'related', 'processes', 'financial', 'sales',
        'customer', 'client', 'manufacturing', 'electrical', 'senior', 'training',
        'services', 'projects', 'solutions', 'information', 'performance',
        'processes', 'solutions', 'management', 'analyst'
    ]

    # Combine tech terms to preserve
    preserved_terms = all_tech_terms + tech_abbreviations

    # Remove tech terms from stopwords
    filtered_stopwords = standard_stopwords - set(preserved_terms)

    # Add exclusion terms to stopwords
    custom_stopwords = list(filtered_stopwords) + exclusion_terms

    # First create a CountVectorizer to find common CS/DS terms
    print("Identifying common technical terms...")
    count_vectorizer = CountVectorizer(
        max_features=2000,
        min_df=10,  # Appears in at least 10 documents
        max_df=0.7,  # Appears in at most 70% of documents
        stop_words=custom_stopwords,
        ngram_range=(1, 2)
    )

    # Get term counts
    count_matrix = count_vectorizer.fit_transform(combined_text)
    term_names = count_vectorizer.get_feature_names_out()
    term_counts = count_matrix.sum(axis=0).A1

    # Find common technical terms - terms that appear more than threshold times
    min_term_count = 30  # Adjust based on your dataset size
    common_tech_terms = [term for term, count in zip(term_names, term_counts)
                         if count > min_term_count]

    print(f"Found {len(common_tech_terms)} potential technical terms")

    # Now create TF-IDF representation focusing on technical terms
    print("Creating TF-IDF representation...")
    tfidf_vectorizer = TfidfVectorizer(
        max_features=5000,
        min_df=5,  # Lower min_df to catch rarer technical terms
        max_df=0.9,
        stop_words=custom_stopwords,
        ngram_range=(1, 3)  # Include trigrams for technical phrases
    )

    # Fit and transform
    tfidf_matrix = tfidf_vectorizer.fit_transform(combined_text)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    print(f"Created TF-IDF matrix with {tfidf_matrix.shape[1]} features")

    # Calculate term importance metrics with technical focus
    doc_freq = np.bincount(tfidf_matrix.nonzero()[1], minlength=len(feature_names))
    avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).flatten()
    tfidf_variance = np.asarray(tfidf_matrix.multiply(tfidf_matrix).mean(axis=0)).flatten() - np.power(avg_tfidf, 2)

    # Give bonus to terms that appear in our technical terms lists
    tech_term_bonus = np.zeros(len(feature_names))
    for i, term in enumerate(feature_names):
        # Check if term is a specific technical term (languages, frameworks, etc.)
        if term in specific_tech_terms:
            tech_term_bonus[i] = 5.0  # Highest bonus for specific technical terms
        # Check if term is a core technical domain term
        elif term in core_tech_terms:
            tech_term_bonus[i] = 3.0  # High bonus for core technical terms
        # Check if term contains technical terms
        else:
            # Check if term contains technical terms
            term_parts = term.split()

            # Check for specific tech terms in the phrase
            tech_word_count = 0
            for tech_term in specific_tech_terms:
                if tech_term in term_parts:
                    tech_word_count += 1

            # Check for core tech terms in the phrase
            core_tech_count = 0
            for tech_term in core_tech_terms:
                if tech_term in term_parts:
                    core_tech_count += 1

            # Calculate the bonus based on contained terms
            if tech_word_count > 0:
                tech_term_bonus[i] = 2.0 + (tech_word_count * 0.5)  # Bonus for containing specific tech words
            elif core_tech_count > 0:
                tech_term_bonus[i] = 1.5 + (core_tech_count * 0.3)  # Bonus for containing core tech words
            # Check if term is in the common terms from corpus
            elif term in common_tech_terms:
                tech_term_bonus[i] = 1.0

    # Combined importance score with technical bonus
    # Multiply by (1 + tech_term_bonus) to give higher weight to technical terms
    importance_score = (doc_freq * tfidf_variance) * (1 + tech_term_bonus)

    # Create dataframe with term importance metrics
    keywords_df = pd.DataFrame({
        'term': feature_names,
        'doc_frequency': doc_freq,
        'avg_tfidf': avg_tfidf,
        'tfidf_variance': tfidf_variance,
        'tech_bonus': tech_term_bonus,
        'importance_score': importance_score
    })

    # Sort by importance score
    keywords_df = keywords_df.sort_values('importance_score', ascending=False)

    print(f"Extracted {len(keywords_df)} terms with improved technical focus")

    return keywords_df, tfidf_matrix, tfidf_vectorizer, common_tech_terms

def main():
    """Main function to load data and extract technical keywords"""
    import os
    print("Current working directory:", os.getcwd())

    # Define file paths
    input_file = '../data/processed/csds_filtered_clean.csv'
    output_dir = 'clustering/output'

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Load the dataset
    print(f"Loading data from {input_file}...")
    df = pd.read_csv(input_file)

    # Check if required columns exist
    required_cols = ['title', 'description', 'skills_desc']
    for col in required_cols:
        if col not in df.columns:
            print(f"Error: Required column '{col}' not found in the dataset.")
            return

    # Extract technical keywords with enhanced method
    keywords_df, tfidf_matrix, vectorizer, common_tech_terms = extract_technical_keywords(df)

    # Save the keywords to CSV
    output_file = os.path.join(output_dir, 'technical_keywords_from_postings.csv')
    keywords_df.to_csv(output_file, index=False)
    print(f"Technical keyword data saved to {output_file}")

    # Save common tech terms for reference
    tech_terms_file = os.path.join(output_dir, 'common_tech_terms.txt')
    with open(tech_terms_file, 'w') as f:
        for term in common_tech_terms:
            f.write(f"{term}\n")
    print(f"Common technical terms saved to {tech_terms_file}")

    # Print top keywords
    print("\nTop 50 technical keywords by importance:")
    for i, (term, score) in enumerate(zip(keywords_df['term'].head(50),
                                          keywords_df['importance_score'].head(50))):
        print(f"{i + 1}. {term} (score: {score:.4f})")


if __name__ == "__main__":
    main()




