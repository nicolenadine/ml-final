# job_vectorizer.py

import os
import pandas as pd
import numpy as np
import pickle
from term_utils import create_term_vector
from custom_terms import get_all_terms


class JobVectorizer:
    """
    Processes job postings into term vectors and manages the vector database.

    This class handles the creation and storage of term vectors for job postings,
    allowing for efficient similarity matching with resume vectors.
    """

    def __init__(self, job_data_path=None, vector_save_path=None):
        """
        Initialize the JobVectorizer.

        Parameters:
        -----------
        job_data_path : str, optional
            Path to the CSV file containing job postings
        vector_save_path : str, optional
            Path to save/load the job vectors database
        """
        self.job_data_path = job_data_path or os.path.join("data", "processed", "csds_filtered_clean.csv")
        self.vector_save_path = vector_save_path or os.path.join("data", "vectors", "job_vectors.pkl")
        self.terms = get_all_terms()
        self.job_vectors = {}  # Map of job_id to vector
        self.job_data = None

    def load_job_data(self):
        """Load job posting data from CSV"""
        if not os.path.exists(self.job_data_path):
            raise FileNotFoundError(f"Job data file not found: {self.job_data_path}")

        print(f"Loading job data from {self.job_data_path}")
        self.job_data = pd.read_csv(self.job_data_path)
        print(f"Loaded {len(self.job_data)} job postings")

        # Check required columns exist
        required_cols = ["job_id", "title", "description"]
        for col in required_cols:
            if col not in self.job_data.columns:
                raise ValueError(f"Required column '{col}' not found in job data")

        return self.job_data

    def vectorize_all_jobs(self, force_rebuild=False):
        """
        Create term vectors for all job postings.

        Parameters:
        -----------
        force_rebuild : bool
            If True, rebuild vectors even if they exist

        Returns:
        --------
        dict
            Dictionary mapping job_id to term vector
        """
        # Check if vectors already exist and can be loaded
        if not force_rebuild and os.path.exists(self.vector_save_path):
            try:
                print(f"Loading existing job vectors from {self.vector_save_path}")
                with open(self.vector_save_path, 'rb') as f:
                    self.job_vectors = pickle.load(f)
                print(f"Loaded vectors for {len(self.job_vectors)} jobs")
                return self.job_vectors
            except Exception as e:
                print(f"Error loading existing vectors: {e}")
                print("Will rebuild vectors")

        # Make sure job data is loaded
        if self.job_data is None:
            self.load_job_data()

        # Create directory for vectors if it doesn't exist
        os.makedirs(os.path.dirname(self.vector_save_path), exist_ok=True)

        # Create vectors for each job
        print(f"Creating vectors for {len(self.job_data)} job postings...")
        self.job_vectors = {}

        for idx, row in self.job_data.iterrows():
            job_id = row['job_id']

            # Combine title and description for better matching
            job_text = f"{row['title']} {row['description']}"
            if isinstance(row.get('skills_desc'), str):
                job_text += f" {row['skills_desc']}"

            # Create vector
            vector = create_term_vector(job_text, self.terms)
            self.job_vectors[job_id] = vector

            # Print progress periodically
            if (idx + 1) % 1000 == 0 or idx == len(self.job_data) - 1:
                print(f"Processed {idx + 1}/{len(self.job_data)} job postings")

        # Save vectors
        with open(self.vector_save_path, 'wb') as f:
            pickle.dump(self.job_vectors, f)
        print(f"Saved vectors to {self.vector_save_path}")

        return self.job_vectors

    def get_job_details(self, job_ids):
        """
        Get details for specified job IDs.

        Parameters:
        -----------
        job_ids : list
            List of job IDs to retrieve details for

        Returns:
        --------
        pandas.DataFrame
            DataFrame containing details for the specified jobs
        """
        if self.job_data is None:
            self.load_job_data()

        return self.job_data[self.job_data['job_id'].isin(job_ids)]

    def get_job_vector(self, job_id):
        """Get vector for a specific job ID"""
        if not self.job_vectors:
            self.vectorize_all_jobs()

        return self.job_vectors.get(job_id)


# Example usage
if __name__ == "__main__":
    # Initialize vectorizer
    vectorizer = JobVectorizer()

    # Load job data and create vectors
    job_vectors = vectorizer.vectorize_all_jobs(force_rebuild=False)

    # Print some stats
    print(f"Total job vectors: {len(job_vectors)}")

    # Print a sample vector
    sample_job_id = list(job_vectors.keys())[0]
    vector = job_vectors[sample_job_id]
    nonzero_terms = sum(1 for v in vector if v > 0)
    print(f"Sample job ID {sample_job_id} has {nonzero_terms} non-zero terms out of {len(vector)}")

    # Get details for sample job
    job_details = vectorizer.get_job_details([sample_job_id])
    print(f"Sample job title: {job_details.iloc[0]['title']}")