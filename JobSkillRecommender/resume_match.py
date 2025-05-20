# match_resume.py

import os
import argparse
import json
from resume_job_system import ResumeJobSystem


def main():
    """
    Command-line script for matching resumes to jobs using KNN.
    """
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description='Match resumes to job postings using KNN machine learning',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Input arguments
    parser.add_argument('resume_path', type=str, help='Path to resume PDF file')
    parser.add_argument('--text', action='store_true', help='Treat resume_path as text file instead of PDF')

    # KNN model parameters
    parser.add_argument('--neighbors', type=int, default=10, help='Number of neighbors for KNN')
    parser.add_argument('--metric', type=str, default='cosine',
                        choices=['cosine', 'euclidean', 'manhattan'],
                        help='Distance metric for KNN')
    parser.add_argument('--algorithm', type=str, default='auto',
                        choices=['auto', 'ball_tree', 'kd_tree', 'brute'],
                        help='Algorithm for KNN')

    # Dimensionality reduction
    parser.add_argument('--dim-reduction', action='store_true', help='Use dimensionality reduction')
    parser.add_argument('--reduction-method', type=str, default='pca', choices=['pca', 'svd'],
                        help='Method for dimensionality reduction')
    parser.add_argument('--components', type=int, default=100, help='Number of components for dimensionality reduction')
    parser.add_argument('--standardize', action='store_true', help='Standardize features')

    # Output options
    parser.add_argument('--top', type=int, default=5, help='Number of top matches to return')
    parser.add_argument('--output', type=str, default=None, help='Output file path for results')
    parser.add_argument('--format', type=str, default='csv', choices=['csv', 'json'],
                        help='Output format for results')

    # System options
    parser.add_argument('--rebuild', action='store_true', help='Force rebuild of job vectors and models')
    parser.add_argument('--optimize', action='store_true', help='Perform hyperparameter optimization')
    parser.add_argument('--info', action='store_true', help='Print system information and exit')
    parser.add_argument('--verbose', action='store_true', help='Print verbose output')

    args = parser.parse_args()

    # Initialize system with KNN parameters
    system = ResumeJobSystem(
        model_params={
            'n_neighbors': args.neighbors,
            'algorithm': args.algorithm,
            'metric': args.metric,
            'use_dim_reduction': args.dim_reduction,
            'reduction_method': args.reduction_method,
            'n_components': args.components,
            'standardize': args.standardize
        }
    )

    # If just requesting info, print and exit
    if args.info:
        system.setup(verbose=args.verbose)
        info = system.get_system_info()
        print(json.dumps(info, indent=2))
        return 0

    # Check if resume file exists
    if not os.path.exists(args.resume_path):
        print(f"Error: File not found at {args.resume_path}")
        return 1

    # Check if it's a PDF (unless text mode)
    if not args.text and not args.resume_path.lower().endswith('.pdf'):
        print("Error: Resume file must be a PDF unless --text is specified")
        return 1

    try:
        # Set up the system
        system.setup(force_rebuild=args.rebuild, optimize=args.optimize, verbose=args.verbose)

        # Match resume to jobs
        if args.text:
            # Read text file
            with open(args.resume_path, 'r', encoding='utf-8') as f:
                resume_text = f.read()

            # Match resume text
            matches = system.match_resume_text(
                resume_text=resume_text,
                top_n=args.top,
                verbose=args.verbose
            )
        else:
            # Match resume file
            matches = system.match_resume(
                resume_path=args.resume_path,
                top_n=args.top,
                verbose=args.verbose
            )

        # Print results
        system.print_matches(matches)

        # Save results if output path specified
        if args.output:
            output_path = args.output
        else:
            # Generate output filename from resume name
            resume_name = os.path.basename(args.resume_path)
            output_path = system.save_match_results(
                matches=matches,
                resume_name=resume_name,
                format=args.format
            )
            print(f"\nMatch results saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())