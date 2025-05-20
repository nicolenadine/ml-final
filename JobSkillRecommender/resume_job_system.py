# resume_job_system.py

import os
import pandas as pd
import time
import pickle
import json
from resume_parser import ResumeParser
from custom_terms import get_all_terms
from term_utils import create_term_vector
from job_vectorizer import JobVectorizer
from knn_matcher import KNNJobMatcher


class ResumeJobSystem:
    """
    Complete system for matching resumes to job postings using KNN.

    This system integrates all components:
    1. Resume parsing (PDF extraction)
    2. Term-based feature engineering
    3. KNN machine learning for job matching
    4. Results processing and visualization
    """

    def __init__(self, data_dir=None, model_params=None):
        """
        Initialize the resume-job matching system.

        Parameters:
        -----------
        data_dir : str, optional
            Base directory for data files
        model_params : dict, optional
            Parameters for the KNN model. If None, uses defaults.
        """
        # Check if running in App Engine
        if os.getenv('GAE_ENV', '').startswith('standard'):
            # Use /tmp for writable operations in App Engine
            self.data_dir = '/tmp'
            self.results_dir = '/tmp/results'

            # Create necessary directories in /tmp
            os.makedirs(os.path.join(self.data_dir, "processed"), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "vectors"), exist_ok=True)
            os.makedirs(os.path.join(self.data_dir, "models"), exist_ok=True)
            os.makedirs(self.results_dir, exist_ok=True)

            # Define paths for App Engine
            job_data_path = os.path.join(self.data_dir, "processed", "csds_filtered_clean.csv")
            vector_save_path = os.path.join(self.data_dir, "vectors", "job_vectors.pkl")

            # Check if we need to copy or create sample data
            if not os.path.exists(job_data_path):
                # Try to copy from the deployment directory first
                try:
                    import shutil
                    source_path = os.path.join("data", "processed", "csds_filtered_clean.csv")
                    if os.path.exists(source_path):
                        shutil.copy(source_path, job_data_path)
                        print(f"Copied data file from {source_path} to {job_data_path}")
                except Exception as e:
                    print(f"Could not copy data file: {e}")
                    # We'll let JobVectorizer handle the missing file with create_sample_data
        else:
            # Local environment - use specified or default directories
            self.data_dir = data_dir or os.path.join(os.getcwd(), "data")
            self.results_dir = os.path.join(self.data_dir, "results")

            # Define paths for local environment
            job_data_path = os.path.join(self.data_dir, "processed", "csds_filtered_clean.csv")
            vector_save_path = os.path.join(self.data_dir, "vectors", "job_vectors.pkl")

            # Create results directory
            os.makedirs(self.results_dir, exist_ok=True)

        # Set up model parameters
        self.model_params = {
            'n_neighbors': 10,
            'algorithm': 'auto',
            'metric': 'cosine',
            'use_dim_reduction': True,
            'reduction_method': 'pca',  # Using PCA as the default
            'n_components': 100,
            'standardize': False
        }

        # Update with provided parameters
        if model_params:
            self.model_params.update(model_params)

        # Initialize components
        self.terms = get_all_terms()
        self.resume_parser = ResumeParser(ocr_enabled=True)

        # Initialize job vectorizer with proper paths
        self.job_vectorizer = JobVectorizer(
            job_data_path=job_data_path,
            vector_save_path=vector_save_path
        )

        # Initialize KNN matcher
        self.knn_matcher = KNNJobMatcher(
            job_vectorizer=self.job_vectorizer,
            **self.model_params
        )



    def setup(self, force_rebuild=False, optimize=False, verbose=True):
        """
        Set up the system by loading data and building models.

        Parameters:
        -----------
        force_rebuild : bool
            If True, force rebuild of vectors and models
        optimize : bool
            If True, perform hyperparameter optimization
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        self
            For method chaining
        """
        # Load job data
        if self.job_vectorizer.job_data is None:
            self.job_vectorizer.load_job_data()

        # Build KNN model
        self.knn_matcher.build_model(force_rebuild=force_rebuild, optimize=optimize, verbose=verbose)

        return self

    def parse_resume(self, resume_path, verbose=False):
        """
        Parse a resume PDF file.

        Parameters:
        -----------
        resume_path : str
            Path to the resume PDF file
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        dict
            Parsed resume data
        """
        start_time = time.time()

        if verbose:
            print(f"Parsing resume: {resume_path}")

        # Check if file exists
        if not os.path.exists(resume_path):
            raise FileNotFoundError(f"Resume file not found: {resume_path}")

        # Parse the resume
        resume_data = self.resume_parser.parse(resume_path)

        if verbose:
            print(f"Extracted {resume_data['word_count']} words using {resume_data['extraction_method']}")
            print(f"Parsing took {time.time() - start_time:.2f} seconds")

        return resume_data

    def match_resume(self, resume_path, top_n=5, explain=True, verbose=False):
        """
        Match a resume file to jobs.

        Parameters:
        -----------
        resume_path : str
            Path to the resume PDF file
        top_n : int
            Number of top matches to return
        explain : bool
            If True, return explanation of match
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        list
            List of tuples with match results
        """
        # Make sure the system is set up
        self.setup(verbose=verbose)

        # Parse the resume
        resume_data = self.parse_resume(resume_path, verbose=verbose)
        resume_text = resume_data.get("text", "")

        if not resume_text:
            raise ValueError("Failed to extract text from resume")

        # Match using the KNN matcher
        matches = self.knn_matcher.match_resume(
            resume_text=resume_text,
            top_n=top_n,
            explain=explain,
            verbose=verbose
        )

        return matches

    def match_resume_text(self, resume_text, top_n=5, explain=True, verbose=False):
        """
        Match resume text to jobs without parsing a file.

        Parameters:
        -----------
        resume_text : str
            Resume text content
        top_n : int
            Number of top matches to return
        explain : bool
            If True, return explanation of match
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        list
            List of tuples with match results
        """
        # Make sure the system is set up
        self.setup(verbose=verbose)

        # Match using the KNN matcher
        matches = self.knn_matcher.match_resume(
            resume_text=resume_text,
            top_n=top_n,
            explain=explain,
            verbose=verbose
        )

        return matches

    def save_match_results(self, matches, resume_name=None, format='csv'):
        """
        Save match results to a file.

        Parameters:
        -----------
        matches : list
            Match results from match_resume or match_resume_text
        resume_name : str, optional
            Name to use in the output file
        format : str
            Output format ('csv', 'json')

        Returns:
        --------
        str
            Path to the saved file
        """
        # Generate filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        resume_prefix = f"{resume_name.replace('.pdf', '')}_" if resume_name else ""
        filename = f"{resume_prefix}matches_{timestamp}.{format}"
        output_path = os.path.join(self.results_dir, filename)

        # Create DataFrame from matches
        if len(matches) > 0 and len(matches[0]) > 3:
            # Matches include explanation
            data = []
            for job_id, similarity, job_details, term_contributions in matches:
                row = {"job_id": job_id, "similarity_score": similarity}
                # Add job details
                for key, value in job_details.items():
                    row[key] = value
                # Add top terms as a string
                row["top_matching_terms"] = ", ".join([term for term, _ in term_contributions[:10]])
                data.append(row)
        else:
            # Matches without explanation
            data = []
            for job_id, similarity, job_details in matches:
                row = {"job_id": job_id, "similarity_score": similarity}
                # Add job details
                for key, value in job_details.items():
                    row[key] = value
                data.append(row)

        # Create DataFrame
        df = pd.DataFrame(data)

        # Save to file
        if format == 'csv':
            df.to_csv(output_path, index=False)
        elif format == 'json':
            df.to_json(output_path, orient='records', indent=4)
        else:
            raise ValueError(f"Unknown format: {format}")

        return output_path

    def print_matches(self, matches, show_terms=True):
        """
        Print formatted match results.

        Parameters:
        -----------
        matches : list
            Match results from match_resume or match_resume_text
        show_terms : bool
            Whether to show the top matching terms
        """
        print("\n===== TOP JOB MATCHES =====")

        # Check if matches include explanation
        has_explanation = len(matches) > 0 and len(matches[0]) > 3

        for i, match_data in enumerate(matches):
            if has_explanation:
                job_id, similarity, job_details, term_contributions = match_data
            else:
                job_id, similarity, job_details = match_data
                term_contributions = []

            print(f"\n[{i + 1}] Match Score: {similarity:.4f} - Job ID: {job_id}")
            print(f"Title: {job_details['title']}")
            print(f"Company: {job_details['company_name']}")

            if 'location' in job_details:
                print(f"Location: {job_details['location']}")

            if 'remote_allowed' in job_details:
                remote = "Yes" if job_details['remote_allowed'] else "No"
                print(f"Remote: {remote}")

            if 'min_salary' in job_details and 'max_salary' in job_details:
                if pd.notna(job_details['min_salary']) and pd.notna(job_details['max_salary']):
                    print(f"Salary Range: ${job_details['min_salary']:,.0f} - ${job_details['max_salary']:,.0f}")

            if 'description' in job_details:
                print("\nDescription (excerpt):")
                desc = job_details['description']
                print(desc[:200] + "..." if len(desc) > 200 else desc)

            if show_terms and has_explanation and term_contributions:
                print("\nTop matching terms:")
                for term, score in term_contributions[:5]:
                    print(f"  - {term}: {score:.4f}")

            print("-" * 50)

    def get_system_info(self):
        """
        Get information about the system.

        Returns:
        --------
        dict
            System information
        """
        # Get KNN model info
        knn_info = self.knn_matcher.get_model_info()

        # Get job data info
        job_data_info = {
            'num_jobs': len(self.job_vectorizer.job_data) if self.job_vectorizer.job_data is not None else 0,
            'job_data_path': self.job_vectorizer.job_data_path,
            'vector_save_path': self.job_vectorizer.vector_save_path
        }

        # Get term info
        term_info = {
            'num_terms': len(self.terms),
            'term_categories': len(getattr(self.terms, 'categories', []))
        }

        return {
            'knn_model': knn_info,
            'job_data': job_data_info,
            'terms': term_info,
            'system_time': time.strftime("%Y-%m-%d %H:%M:%S")
        }


# Example usage
if __name__ == "__main__":
    # Initialize the system
    system = ResumeJobSystem(
        model_params={
            'algorithm': 'auto',
            'metric': 'cosine',
            'use_svd': True,
            'n_components': 100
        }
    )

    # Setup the system
    system.setup(verbose=True)

    # Print system info
    print("\nSystem Info:")
    info = system.get_system_info()
    print(json.dumps(info, indent=2))

    # Sample resume text for testing
    sample_resume = """
    SENIOR DATA SCIENTIST

    Professional Summary
    Experienced Data Scientist with 7+ years of experience in developing machine learning models, 
    statistical analysis, and data visualization. Strong background in Python, R, and SQL with 
    expertise in deep learning and natural language processing.

    Skills
    • Programming: Python, R, SQL, Java
    • Machine Learning: TensorFlow, PyTorch, scikit-learn, XGBoost
    • Data Analysis: Pandas, NumPy, SciPy, Statsmodels
    • Big Data: Spark, Hadoop, Hive
    • Cloud: AWS, Azure, GCP
    • Visualization: Tableau, PowerBI, Matplotlib

    Experience

    Senior Data Scientist
    ABC Tech Solutions, San Francisco, CA
    January 2020 - Present
    • Led development of customer churn prediction model with 85% accuracy
    • Built NLP pipeline for sentiment analysis of customer feedback
    • Implemented time series forecasting for inventory optimization
    • Mentored junior data scientists and collaborated with cross-functional teams

    Data Scientist
    XYZ Analytics, Seattle, WA
    June 2017 - December 2019
    • Created recommendation engine that increased user engagement by 25%
    • Developed ETL pipelines for processing large datasets
    • Built interactive dashboards for business KPIs
    • Performed A/B testing to optimize marketing campaigns

    Education
    Master of Science in Data Science
    University of Washington, 2017

    Bachelor of Science in Computer Science
    University of California, Berkeley, 2015
    """

    # Match resume to jobs
    matches = system.match_resume_text(sample_resume, top_n=5, verbose=True)

    # Print results
    system.print_matches(matches)

    # Save results
    output_path = system.save_match_results(matches, resume_name="sample_resume", format='csv')
    print(f"\nMatch results saved to: {output_path}")