# knn_matcher.py

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import pickle
import os
import time
from joblib import dump, load
from job_vectorizer import JobVectorizer
from term_utils import create_term_vector
from custom_terms import get_all_terms


class KNNJobMatcher:
    """
    Job matching using k-nearest neighbors algorithm with scikit-learn.

    This class implements a machine learning approach to resume-job matching
    using the k-nearest neighbors algorithm. It includes options for:
    - Dimensionality reduction with TruncatedSVD (LSA)
    - Feature standardization
    - Multiple distance metrics
    - Different KNN algorithms for efficient search
    - Hyperparameter optimization
    """

    def __init__(self, job_vectorizer=None, n_neighbors=5, algorithm='auto',
                 metric='euclidean', use_dim_reduction=True, reduction_method='pca', n_components=100,
                 standardize=False):
        """
        Initialize the KNN job matcher.

        Parameters:
        -----------
        job_vectorizer : JobVectorizer, optional
            Pre-initialized job vectorizer. If None, a new one will be created.
        n_neighbors : int
            Number of neighbors to find in the KNN algorithm
        algorithm : str
            Algorithm used to compute nearest neighbors
            ('ball_tree', 'kd_tree', 'brute', 'auto')
        metric : str
            Distance metric to use ('cosine', 'euclidean', 'manhattan', etc.)
        use_dim_reduction : bool
            Whether to use dimensionality reduction
        reduction_method : str
            Method to use for dimensionality reduction ('pca' or 'svd')
        n_components : int
            Number of components to keep in dimensionality reduction
        standardize : bool
            Whether to standardize features before KNN
        """
        self.job_vectorizer = job_vectorizer or JobVectorizer()
        self.n_neighbors = n_neighbors
        self.algorithm = algorithm
        self.metric = metric
        self.use_dim_reduction = use_dim_reduction
        self.reduction_method = reduction_method
        self.n_components = n_components
        self.standardize = standardize

        self.terms = get_all_terms()
        self.knn_model = None
        self.pipeline = None
        self.job_ids = None
        self.vectors_array = None

        # Paths for model files
        self.model_dir = os.path.join("data", "models")
        self.knn_model_path = os.path.join(self.model_dir, "knn_model.joblib")
        self.pipeline_path = os.path.join(self.model_dir, "pipeline.joblib")
        self.job_ids_path = os.path.join(self.model_dir, "job_ids.pkl")
        self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")

    def build_pipeline(self):
        """
        Build the scikit-learn pipeline for preprocessing and KNN.

        Returns:
        --------
        sklearn.pipeline.Pipeline
            Preprocessing and KNN pipeline
        """
        steps = []

        # Add standardization if enabled (should come before dim reduction)
        if self.standardize:
            steps.append(('scaler', StandardScaler()))

        # Add dimensionality reduction if enabled
        if self.use_dim_reduction:
            if self.reduction_method == 'pca':
                steps.append(('reducer', PCA(
                    n_components=min(self.n_components, len(self.terms) - 1),
                    random_state=42
                )))
            elif self.reduction_method == 'svd':
                steps.append(('reducer', TruncatedSVD(
                    n_components=min(self.n_components, len(self.terms) - 1),
                    random_state=42,
                    n_iter=7
                )))

        # Add KNN model
        steps.append(('knn', NearestNeighbors(
            n_neighbors=self.n_neighbors,
            algorithm=self.algorithm,
            metric=self.metric
        )))

        # Create pipeline
        return Pipeline(steps)

    def optimize_hyperparameters(self, X, verbose=True):
        """
        Perform hyperparameter optimization for the KNN model.

        Parameters:
        -----------
        X : numpy.ndarray
            Feature vectors to use for optimization
        verbose : bool
            Whether to print optimization progress

        Returns:
        --------
        dict
            Best hyperparameters
        """
        if verbose:
            print("Performing hyperparameter optimization...")

        # Define the pipeline for optimization
        pipeline = Pipeline([
            ('reducer', PCA(random_state=42)),
            ('knn', NearestNeighbors())
        ])

        # Define parameter grid
        param_grid = {
            'reducer__n_components': [50, 100, 200],
            'knn__n_neighbors': [5, 10, 20],
            'knn__algorithm': ['auto', 'ball_tree', 'kd_tree'],
            'knn__metric': ['cosine', 'euclidean']
        }

        # Create grid search
        grid_search = GridSearchCV(
            pipeline,
            param_grid,
            cv=3,
            verbose=2 if verbose else 0
        )

        # Fit grid search
        grid_search.fit(X)

        # Update parameters with best values
        self.n_components = grid_search.best_params_['reducer__n_components']
        self.n_neighbors = grid_search.best_params_['knn__n_neighbors']
        self.algorithm = grid_search.best_params_['knn__algorithm']
        self.metric = grid_search.best_params_['knn__metric']

        if verbose:
            print(f"Best parameters: {grid_search.best_params_}")

        return grid_search.best_params_

    def build_model(self, force_rebuild=False, optimize=False, verbose=True):
        """
        Build the KNN model using job vectors.

        Parameters:
        -----------
        force_rebuild : bool
            If True, rebuild the model even if it exists
        optimize : bool
            If True, perform hyperparameter optimization
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        self
            For method chaining
        """
        # Determine if running in App Engine
        if os.getenv('GAE_ENV', '').startswith('standard'):
            # Use temp directory for model files
            self.model_dir = '/tmp/models'
            self.knn_model_path = os.path.join(self.model_dir, "knn_model.joblib")
            self.pipeline_path = os.path.join(self.model_dir, "pipeline.joblib")
            self.job_ids_path = os.path.join(self.model_dir, "job_ids.pkl")
            self.metadata_path = os.path.join(self.model_dir, "model_metadata.json")


        # Try to load existing model if available
        if not force_rebuild and os.path.exists(self.knn_model_path) and os.path.exists(self.pipeline_path):
            try:
                if verbose:
                    print(f"Loading existing KNN model from {self.knn_model_path}")

                # Load KNN model
                self.knn_model = load(self.knn_model_path)

                # Load pipeline
                self.pipeline = load(self.pipeline_path)

                # Load job IDs
                with open(self.job_ids_path, 'rb') as f:
                    self.job_ids = pickle.load(f)

                # Load metadata
                metadata = pd.read_json(self.metadata_path, orient='records', lines=True).iloc[0].to_dict()
                self.n_neighbors = metadata['n_neighbors']
                self.algorithm = metadata['algorithm']
                self.metric = metadata['metric']
                self.use_dim_reduction = metadata.get('use_dim_reduction', metadata.get('use_svd', True))
                self.reduction_method = metadata.get('reduction_method', 'pca')
                self.n_components = metadata['n_components']
                self.standardize = metadata['standardize']

                if verbose:
                    print(f"Model loaded with {len(self.job_ids)} job vectors")

                return self
            except Exception as e:
                if verbose:
                    print(f"Error loading existing model: {e}")
                    print("Will rebuild model")

        # Make sure job vectors are loaded
        start_time = time.time()
        job_vectors = self.job_vectorizer.vectorize_all_jobs(force_rebuild=False)

        # Convert dictionary of job vectors to a numpy array
        job_ids = list(job_vectors.keys())
        vectors = np.array([job_vectors[job_id] for job_id in job_ids])
        self.vectors_array = vectors
        self.job_ids = job_ids

        if verbose:
            print(f"Prepared {len(vectors)} job vectors with {vectors.shape[1]} features")
            print(f"Vector preparation took {time.time() - start_time:.2f} seconds")

        # Perform hyperparameter optimization if requested
        if optimize:
            best_params = self.optimize_hyperparameters(vectors, verbose=verbose)

        # Build pipeline
        model_start_time = time.time()
        self.pipeline = self.build_pipeline()

        # Fit the pipeline
        if verbose:
            print(f"Building KNN model with {len(vectors)} job vectors...")
            print(f"Pipeline steps: {[step[0] for step in self.pipeline.steps]}")

        self.pipeline.fit(vectors)

        # Get the KNN model from the pipeline
        self.knn_model = self.pipeline.named_steps['knn']

        if verbose:
            print(f"Model building took {time.time() - model_start_time:.2f} seconds")

        # Save the model
        os.makedirs(self.model_dir, exist_ok=True)

        # Save KNN model
        dump(self.knn_model, self.knn_model_path)

        # Save pipeline
        dump(self.pipeline, self.pipeline_path)

        # Save job IDs
        with open(self.job_ids_path, 'wb') as f:
            pickle.dump(self.job_ids, f)

        # Save metadata
        metadata = pd.DataFrame([{
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'use_dim_reduction': self.use_dim_reduction,
            'reduction_method': self.reduction_method,
            'n_components': self.n_components,
            'standardize': self.standardize,
            'num_jobs': len(self.job_ids),
            'num_features': vectors.shape[1],
            'build_time': time.time() - start_time,
            'date_built': pd.Timestamp.now().isoformat()
        }])
        metadata.to_json(self.metadata_path, orient='records', lines=True)

        if verbose:
            print(f"Saved model to {self.model_dir}")

        return self

    def get_model_info(self):
        """
        Get information about the KNN model.

        Returns:
        --------
        dict
            Model information
        """
        # Load metadata if available
        if os.path.exists(self.metadata_path):
            metadata = pd.read_json(self.metadata_path, orient='records', lines=True).iloc[0].to_dict()
            return metadata

        # Otherwise, return current settings
        return {
            'n_neighbors': self.n_neighbors,
            'algorithm': self.algorithm,
            'metric': self.metric,
            'use_dim_reduction': self.use_dim_reduction,  # Fixed from use_svd
            'reduction_method': self.reduction_method,
            'n_components': self.n_components,
            'standardize': self.standardize,
            'num_jobs': len(self.job_ids) if self.job_ids else 0,
            'model_built': self.knn_model is not None
        }


    def transform_resume_vector(self, resume_vector):
        """
        Transform a resume vector using the same pipeline as job vectors.

        Parameters:
        -----------
        resume_vector : numpy.ndarray
            Resume feature vector

        Returns:
        --------
        numpy.ndarray
            Transformed resume vector
        """
        # Skip if no preprocessing in pipeline (just KNN)
        if len(self.pipeline.steps) == 1:
            return resume_vector

        # Apply all transformations except KNN
        transformed = resume_vector
        for name, transform in self.pipeline.steps[:-1]:
            transformed = transform.transform(transformed.reshape(1, -1))

        return transformed

    def match_resume(self, resume_text, top_n=None, explain=False, verbose=False):
        """
        Match a resume to jobs using KNN.

        Parameters:
        -----------
        resume_text : str
            Resume text content
        top_n : int, optional
            Number of top matches to return. If None, uses self.n_neighbors
        explain : bool
            If True, return additional information about the match
        verbose : bool
            Whether to print progress information

        Returns:
        --------
        list
            List of tuples (job_id, similarity_score, job_details_dict)
            If explain=True, also includes explanation data
        """
        if self.knn_model is None or self.pipeline is None:
            self.build_model(verbose=verbose)

        # Number of neighbors to find
        n_neighbors = top_n or self.n_neighbors
        n_neighbors = min(n_neighbors, len(self.job_ids))

        # Create resume vector
        start_time = time.time()
        resume_vector = create_term_vector(resume_text, self.terms)

        if verbose:
            print(f"Created resume vector in {time.time() - start_time:.2f} seconds")
            nonzero_terms = sum(1 for v in resume_vector if v > 0)
            print(f"Resume has {nonzero_terms} non-zero terms out of {len(resume_vector)}")

        # Transform resume vector (if pipeline has preprocessing)
        if len(self.pipeline.steps) > 1:
            resume_vector_transformed = self.transform_resume_vector(resume_vector)
        else:
            resume_vector_transformed = resume_vector.reshape(1, -1)

        # Find nearest neighbors
        match_start_time = time.time()
        distances, indices = self.knn_model.kneighbors(
            resume_vector_transformed,
            n_neighbors=n_neighbors
        )

        if verbose:
            print(f"Found nearest neighbors in {time.time() - match_start_time:.2f} seconds")

        # Flatten the results
        distances = distances.flatten()
        indices = indices.flatten()

        # Convert distances to similarity scores based on the metric
        if self.metric == 'cosine':
            # Convert cosine distance to cosine similarity
            similarities = 1 - distances
        elif self.metric in ['jaccard', 'hamming']:
            # For set-based distances, similarity is the inverse of distance
            similarities = 1 - distances
        elif self.metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'canberra']:
            # For distance metrics, normalize to [0, 1] range and invert
            max_distance = np.max(distances) if len(distances) > 0 else 1.0
            if max_distance == 0:
                # Avoid division by zero
                similarities = np.ones_like(distances)
            else:
                similarities = 1 - (distances / max_distance)
        else:
            # Generic fallback for other metrics
            similarities = 1 / (1 + distances)

        # Get matched job IDs
        matched_job_ids = [self.job_ids[idx] for idx in indices]

        # Get job details
        job_details = self.job_vectorizer.get_job_details(matched_job_ids)

        # Prepare results
        results = []
        for i, job_id in enumerate(matched_job_ids):
            # Find this job in the details dataframe
            job_row = job_details[job_details['job_id'] == job_id]
            if len(job_row) > 0:
                # Convert to dict for easier handling
                job_dict = job_row.iloc[0].to_dict()

                # Add explanation data if requested
                if explain:
                    # Get original job vector
                    job_vector = self.job_vectorizer.get_job_vector(job_id)

                    # Calculate term-by-term contribution
                    term_contributions = []
                    for j, term in enumerate(self.terms):
                        # Skip if both are zero
                        if resume_vector[j] == 0 and job_vector[j] == 0:
                            continue

                        # Calculate contribution based on similarity method
                        if self.metric == 'cosine':
                            # For cosine, contribution is the product of the term values
                            contribution = float(resume_vector[j] * job_vector[j])
                        elif self.metric == 'jaccard':
                            # For Jaccard, contribution is based on set intersection/union
                            if resume_vector[j] > 0 and job_vector[j] > 0:
                                contribution = 1.0
                            else:
                                contribution = 0.0
                        elif self.metric == 'hamming':
                            # For Hamming, contribution is 1 if values match, 0 otherwise
                            contribution = 1.0 if resume_vector[j] == job_vector[j] else 0.0
                        elif self.metric in ['euclidean', 'manhattan', 'chebyshev', 'minkowski']:
                            # For distance metrics, negative contribution is the difference
                            if self.metric == 'euclidean':
                                contribution = float(-((resume_vector[j] - job_vector[j]) ** 2))
                            elif self.metric == 'manhattan':
                                contribution = float(-abs(resume_vector[j] - job_vector[j]))
                            elif self.metric == 'chebyshev':
                                contribution = float(-abs(resume_vector[j] - job_vector[j]))
                            else:  # minkowski
                                contribution = float(-abs(resume_vector[j] - job_vector[j]))
                        elif self.metric == 'canberra':
                            # For Canberra, contribution is weighted by the sum of absolute values
                            if resume_vector[j] == 0 and job_vector[j] == 0:
                                contribution = 0.0
                            else:
                                abs_sum = abs(resume_vector[j]) + abs(job_vector[j])
                                if abs_sum > 0:
                                    contribution = float(-abs(resume_vector[j] - job_vector[j]) / abs_sum)
                                else:
                                    contribution = 0.0
                        else:
                            # For other metrics, use a simple heuristic
                            contribution = 1.0 if resume_vector[j] > 0 and job_vector[j] > 0 else 0.0

                        if contribution != 0:
                            term_contributions.append((term, contribution))

                    # Sort by contribution (descending for positive, ascending for negative)
                    if self.metric in ['cosine', 'jaccard', 'hamming']:
                        # Sort by absolute value for metrics where higher is better
                        term_contributions.sort(key=lambda x: x[1], reverse=True)
                    else:
                        # Sort by absolute value for metrics where lower is better
                        term_contributions.sort(key=lambda x: abs(x[1]), reverse=True)

                    # Add to results with explanation
                    results.append((
                        job_id,
                        similarities[i],
                        job_dict,
                        term_contributions[:20]  # Top 20 contributing terms
                    ))
                else:
                    # Add to results without explanation
                    results.append((job_id, similarities[i], job_dict))

        if verbose:
            print(f"Total matching process took {time.time() - start_time:.2f} seconds")

        return results


# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Initialize KNN matcher
    matcher = KNNJobMatcher(
        algorithm='auto',
        metric='euclidean',
        use_svd=True,
        n_components=100
    )

    # Build the model (only needed once)
    matcher.build_model(verbose=True)

    # Print model info
    print("\nModel Info:")
    info = matcher.get_model_info()
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Test with a sample resume
    sample_resume = """
    Data Scientist with 5 years of experience in Python, R, and SQL.
    Skilled in machine learning algorithms, deep learning frameworks, and statistical analysis.
    Proficient in TensorFlow, PyTorch, and scikit-learn.
    Experience with data visualization tools like Tableau and PowerBI.
    Strong background in ETL processes and data pipeline development.
    """

    # Match resume to jobs with explanation
    matches = matcher.match_resume(sample_resume, top_n=5, explain=True, verbose=True)

    # Print results
    print("\n===== TOP JOB MATCHES (KNN) =====")
    for i, match_data in enumerate(matches):
        job_id, similarity, job_details, term_contributions = match_data
        print(f"\n[{i + 1}] Match Score: {similarity:.4f} - Job ID: {job_id}")
        print(f"Title: {job_details['title']}")
        print(f"Company: {job_details['company_name']}")
        if 'location' in job_details:
            print(f"Location: {job_details['location']}")

        print("\nTop contributing terms:")
        for term, score in term_contributions[:5]:
            print(f"  - {term}: {score:.4f}")

        print("-" * 50)

# Example usage
if __name__ == "__main__":
    import pandas as pd

    # Initialize KNN matcher
    matcher = KNNJobMatcher(algorithm='brute', metric='distance')

    # Build the model (only needed once)
    matcher.build_model()

    # Test with a sample resume
    sample_resume = """
    Data Scientist with 5 years of experience in Python, R, and SQL.
    Skilled in machine learning algorithms, deep learning frameworks, and statistical analysis.
    Proficient in TensorFlow, PyTorch, and scikit-learn.
    Experience with data visualization tools like Tableau and PowerBI.
    """

    # Match resume to jobs
    matches = matcher.match_resume(sample_resume, top_n=5)

    # Print results
    print("\n===== TOP JOB MATCHES (KNN) =====")
    for i, (job_id, similarity, job_details) in enumerate(matches):
        print(f"\n[{i + 1}] Match Score: {similarity:.4f} - Job ID: {job_id}")
        print(f"Title: {job_details['title']}")
        print(f"Company: {job_details['company_name']}")
        if 'location' in job_details:
            print(f"Location: {job_details['location']}")
        print("-" * 50)