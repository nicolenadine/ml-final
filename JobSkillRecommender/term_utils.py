"""
Term Utilities for Job-Resume Matching
-------------------------------------
Utility functions for working with custom terms in the job-resume matching system.
"""

import re
import numpy as np
from custom_terms import get_all_terms


def create_term_vector(text, terms=None):
    """
    Create a feature vector from text using the custom terms.

    Parameters:
    -----------
    text : str
        Text content to vectorize (resume or job posting)
    terms : list, optional
        List of terms to use for vectorization. If None, uses all custom terms.

    Returns:
    --------
    numpy.ndarray
        Feature vector with dimensions matching the terms list
    """
    # Get terms if not provided
    if terms is None:
        terms = get_all_terms()

    # Initialize vector
    vector = np.zeros(len(terms))

    # Clean and preprocess text
    text = _preprocess_text(text)

    # Count occurrences of each term
    for i, term in enumerate(terms):
        pattern = r'\b' + re.escape(term.lower()) + r'\b'
        count = len(re.findall(pattern, text))
        vector[i] = count

    return vector


def _preprocess_text(text):
    """
    Preprocess text for better term matching.

    Parameters:
    -----------
    text : str
        Text to preprocess

    Returns:
    --------
    str
        Preprocessed text
    """
    if not isinstance(text, str):
        return ""

    # Convert to lowercase
    text = text.lower()

    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text)

    # Handle some common substitutions for better matching
    replacements = {
        'c++': 'cplusplus',
        'c#': 'csharp',
        '.net': 'dotnet',
        'node.js': 'nodejs',
        'react.js': 'reactjs',
    }

    for original, replacement in replacements.items():
        text = re.sub(r'\b' + re.escape(original) + r'\b', replacement, text)

    return text


def calculate_similarity(vec1, vec2, method='cosine'):
    """
    Calculate similarity between two term vectors.

    Parameters:
    -----------
    vec1 : numpy.ndarray
        First term vector
    vec2 : numpy.ndarray
        Second term vector
    method : str
        Similarity method to use ('cosine', 'jaccard', 'euclidean')

    Returns:
    --------
    float
        Similarity score (higher is more similar)
    """
    if method == 'cosine':
        # Cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0  # Handle zero vectors

        return dot_product / (norm1 * norm2)

    elif method == 'jaccard':
        # Convert to binary vectors for Jaccard similarity
        bin_vec1 = vec1 > 0
        bin_vec2 = vec2 > 0

        intersection = np.logical_and(bin_vec1, bin_vec2).sum()
        union = np.logical_or(bin_vec1, bin_vec2).sum()

        if union == 0:
            return 0

        return intersection / union

    elif method == 'euclidean':
        # Euclidean distance (converted to similarity)
        distance = np.linalg.norm(vec1 - vec2)
        return 1 / (1 + distance)  # Convert distance to similarity

    else:
        raise ValueError(f"Unknown similarity method: {method}")


# Example usage
if __name__ == "__main__":
    # Example resume text
    resume_text = """
    Data Scientist with 3 years of experience in Python, R, and SQL.
    Experienced in machine learning, deep learning, and data visualization.
    Proficient in TensorFlow, PyTorch, and scikit-learn.
    """

    # Create vector
    terms = get_all_terms()
    vector = create_term_vector(resume_text, terms)

    # Print non-zero elements
    print("Terms found in resume:")
    for i, term in enumerate(terms):
        if vector[i] > 0:
            print(f"  - {term}: {vector[i]}")