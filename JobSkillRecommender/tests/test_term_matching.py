"""
Test Term Matching
-----------------
Simple test script to verify that the custom term matching is working correctly.
"""

from custom_terms import get_all_terms
from term_utils import create_term_vector, calculate_similarity


def main():
    """Run a simple test of the term matching functionality"""
    # Get all terms
    terms = get_all_terms()
    print(f"Loaded {len(terms)} unique terms")

    # Example resume and job posting
    resume_text = """
    Data Scientist with 5 years of experience in Python, R, and SQL.
    Skilled in machine learning algorithms, deep learning frameworks, and statistical analysis.
    Proficient in TensorFlow, PyTorch, and scikit-learn.
    Experience with data visualization tools like Tableau and PowerBI.
    Strong background in ETL processes and data pipeline development.
    """

    job_posting = """
    We are seeking a Data Scientist to join our team. The ideal candidate will have:
    - Strong programming skills in Python and SQL
    - Experience with machine learning and deep learning frameworks
    - Knowledge of data visualization and business intelligence tools
    - Familiarity with cloud platforms like AWS or Azure
    - Experience with big data technologies
    """

    # Create vectors
    resume_vector = create_term_vector(resume_text, terms)
    job_vector = create_term_vector(job_posting, terms)

    # Calculate similarity
    similarity = calculate_similarity(resume_vector, job_vector, method='cosine')

    # Print results
    print("\nTerms found in resume:")
    resume_terms = [(term, resume_vector[i]) for i, term in enumerate(terms) if resume_vector[i] > 0]
    resume_terms.sort(key=lambda x: x[1], reverse=True)
    for term, count in resume_terms:
        print(f"  - {term}: {count}")

    print("\nTerms found in job posting:")
    job_terms = [(term, job_vector[i]) for i, term in enumerate(terms) if job_vector[i] > 0]
    job_terms.sort(key=lambda x: x[1], reverse=True)
    for term, count in job_terms:
        print(f"  - {term}: {count}")

    print(f"\nCosine similarity: {similarity:.4f}")

    # Try different similarity measures
    jaccard = calculate_similarity(resume_vector, job_vector, method='jaccard')
    euclidean = calculate_similarity(resume_vector, job_vector, method='euclidean')

    print(f"Jaccard similarity: {jaccard:.4f}")
    print(f"Euclidean similarity: {euclidean:.4f}")


if __name__ == "__main__":
    main()