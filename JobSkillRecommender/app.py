# app.py

import os
import tempfile
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from resume_job_system import ResumeJobSystem
import pandas as pd


def initialize_app_environment():
    """Initialize the application environment, including directories and data files"""
    # Check if running in App Engine
    if os.getenv('GAE_ENV', '').startswith('standard'):
        print("Running in App Engine, setting up environment...")
        # Create necessary directories
        os.makedirs('/tmp/processed', exist_ok=True)
        os.makedirs('/tmp/vectors', exist_ok=True)
        os.makedirs('/tmp/models', exist_ok=True)
        os.makedirs('/tmp/results', exist_ok=True)

        # Try to copy data file
        try:
            import shutil
            data_source = os.path.join("data", "processed", "csds_filtered_clean.csv")
            data_dest = os.path.join("/tmp", "processed", "csds_filtered_clean.csv")
            if os.path.exists(data_source) and not os.path.exists(data_dest):
                shutil.copy(data_source, data_dest)
                print(f"Copied data file to {data_dest}")
        except Exception as e:
            print(f"Warning: Could not copy data file: {e}")

        # Try to copy vectors file
        try:
            vectors_source = os.path.join("data", "vectors", "job_vectors.pkl")
            vectors_dest = os.path.join("/tmp", "vectors", "job_vectors.pkl")
            if os.path.exists(vectors_source) and not os.path.exists(vectors_dest):
                shutil.copy(vectors_source, vectors_dest)
                print(f"Copied vectors file to {vectors_dest}")
        except Exception as e:
            print(f"Warning: Could not copy vectors file: {e}")
    else:
        print("Running in local environment")


# Call this function before initializing Flask
initialize_app_environment()

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Change this in production

# Configure upload folder
if os.getenv('GAE_ENV', '').startswith('standard'):
    UPLOAD_FOLDER = '/tmp'
else:
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

ALLOWED_EXTENSIONS = {'pdf'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize resume-job system with ML model
print("Initializing job matching system...")
job_system = ResumeJobSystem(
    model_params={
        'n_neighbors': 20,  # Use more neighbors than we'll display
        'algorithm': 'auto',
        'metric': 'euclidean',  # Default metric
        'use_dim_reduction': True,
        'reduction_method': 'pca',  # Using PCA instead of SVD
        'n_components': 100
    }
)

# Initialize system on startup
print("Setting up system (checking for cached vectors and models)...")
job_system.setup(verbose=True)
print("Job matching system initialized successfully!")


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Home page with resume upload form"""
    # Get system info
    system_info = job_system.get_system_info()

    # Add formatted info for display
    display_info = {
        'Number of Jobs': system_info['job_data']['num_jobs'],
        'Number of Terms': system_info['terms']['num_terms'],
        'ML Algorithm': f"K-Nearest Neighbors (k={system_info['knn_model']['n_neighbors']})",
        'Distance Metric': system_info['knn_model']['metric'],
        'Dimensionality Reduction': f"PCA ({system_info['knn_model'].get('n_components', 100)} components)"
    }

    return render_template('index.html', system_info=display_info)


@app.route('/upload', methods=['POST'])
def upload_resume():
    """Handle resume upload and matching"""

    if 'resume' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['resume']

    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get number of matches to return
        top_n = int(request.form.get('top_n', 5))

        # Process the resume and match to jobs
        try:
            # Match resume to jobs
            matches = job_system.match_resume(
                resume_path=filepath,
                top_n=top_n,
                explain=True,
                verbose=True
            )

            # Convert to a format easier to use in templates
            results = []
            terms_data = []

            for match_data in matches:
                job_id, similarity, job_details, term_contributions = match_data

                # Clean up job details for display
                cleaned_details = {
                    'title': job_details.get('title', 'N/A'),
                    'company': job_details.get('company_name', 'N/A'),
                    'location': job_details.get('location', 'N/A'),
                    'match_score': f"{similarity:.2f}",
                    'match_percent': f"{similarity * 100:.0f}%",
                    'description': job_details.get('description', 'No description available.'),
                    'url': job_details.get('job_posting_url', '#'),
                    'remote': 'Yes' if job_details.get('remote_allowed') else 'No',
                    'job_id': job_id
                }

                # Add salary information if available
                if 'min_salary' in job_details and 'max_salary' in job_details:
                    if pd.notna(job_details['min_salary']) and pd.notna(job_details['max_salary']):
                        cleaned_details[
                            'salary'] = f"${int(job_details['min_salary']):,} - ${int(job_details['max_salary']):,}"
                    else:
                        cleaned_details['salary'] = 'Not specified'
                else:
                    cleaned_details['salary'] = 'Not specified'

                # Add terms data for the top match
                if len(results) == 0:
                    terms_data = [
                        {"term": term, "score": float(score)}
                        for term, score in term_contributions[:15]
                    ]

                results.append(cleaned_details)

            # Load system info for the results page
            system_info = job_system.get_system_info()

            # Get top terms for word cloud
            top_terms = [term for term, _ in term_contributions[:30]]

            # Convert to JSON for JavaScript
            import json
            terms_data_json = json.dumps(terms_data)
            top_terms_json = json.dumps(top_terms)

            return render_template('results.html',
                                   results=results,
                                   resume_name=filename,
                                   terms_data=terms_data,
                                   top_terms=top_terms,
                                   terms_data_json=terms_data_json,
                                   top_terms_json=top_terms_json,
                                   system_info=system_info)

        except Exception as e:
            flash(f'Error processing resume: {str(e)}')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type. Please upload a PDF.')
        return redirect(url_for('index'))


@app.route('/config', methods=['GET', 'POST'])
def config():
    """Configuration page for the matching system"""
    if request.method == 'POST':
        try:
            # Get parameters from form
            metric = request.form.get('metric', 'cosine')
            n_neighbors = int(request.form.get('n_neighbors', 20))
            algorithm = request.form.get('algorithm', 'auto')
            use_dim_reduction = request.form.get('use_dim_reduction') == 'on'
            reduction_method = request.form.get('reduction_method', 'pca')
            n_components = int(request.form.get('n_components', 100))
            standardize = request.form.get('standardize') == 'on'

            # Validate parameters
            if n_neighbors < 1:
                n_neighbors = 5

            if n_components < 10:
                n_components = 10

            # Define algorithm-metric compatibility
            algorithm_compatibility = {
                'auto': ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'jaccard', 'hamming',
                         'canberra'],
                'ball_tree': ['euclidean', 'manhattan', 'chebyshev', 'minkowski', 'hamming'],
                'kd_tree': ['euclidean', 'manhattan', 'minkowski'],
                'brute': ['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'jaccard', 'hamming',
                          'canberra']
            }

            # Check compatibility
            compatible_metrics = algorithm_compatibility.get(algorithm, ['cosine', 'euclidean'])

            if metric not in compatible_metrics:
                # Create a readable list of compatible metrics for the error message
                compatible_metrics_str = ", ".join([f"'{m}'" for m in compatible_metrics])
                flash(
                    f"Error updating configuration: The '{algorithm}' algorithm is not compatible with the '{metric}' metric. Compatible metrics for this algorithm are: {compatible_metrics_str}.")

                # Keep the current configuration, don't update
                return redirect(url_for('config'))

            # Check if settings have changed
            system_info = job_system.get_system_info()
            current_settings = system_info.get('knn_model', {})

            settings_changed = (
                    current_settings.get('metric') != metric or
                    current_settings.get('n_neighbors') != n_neighbors or
                    current_settings.get('algorithm') != algorithm or
                    current_settings.get('use_dim_reduction') != use_dim_reduction or
                    current_settings.get('reduction_method') != reduction_method or
                    current_settings.get('n_components') != n_components or
                    current_settings.get('standardize') != standardize
            )

            if settings_changed:
                # Update job system parameters
                job_system.knn_matcher.metric = metric
                job_system.knn_matcher.n_neighbors = n_neighbors
                job_system.knn_matcher.algorithm = algorithm
                job_system.knn_matcher.use_dim_reduction = use_dim_reduction
                job_system.knn_matcher.reduction_method = reduction_method
                job_system.knn_matcher.n_components = n_components
                job_system.knn_matcher.standardize = standardize

                # Force rebuild with the new parameters
                job_system.setup(force_rebuild=True, verbose=True)

                flash(
                    f'Configuration updated and model rebuilt with {metric} distance metric and {algorithm} algorithm')
            else:
                flash('No changes detected in configuration')

            return redirect(url_for('index'))

        except Exception as e:
            flash(f'Error updating configuration: {str(e)}')
            return redirect(url_for('config'))

    # GET request - display current configuration
    system_info = job_system.get_system_info()

    return render_template('config.html', system_info=system_info)


@app.route('/api/match', methods=['POST'])
def api_match():
    """API endpoint for resume matching"""
    if 'resume' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['resume']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        # Save to a temporary file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
        file.save(temp_file.name)
        temp_file.close()

        # Get number of matches to return
        top_n = int(request.args.get('top_n', 5))

        try:
            # Match resume to jobs
            matches = job_system.match_resume(
                resume_path=temp_file.name,
                top_n=top_n,
                explain=True
            )

            # Convert to a format easier to use in API
            results = []
            for match_data in matches:
                job_id, similarity, job_details, term_contributions = match_data

                # Prepare result
                result = {
                    'job_id': job_id,
                    'similarity_score': float(similarity),
                    'top_terms': [
                        {"term": term, "score": float(score)}
                        for term, score in term_contributions[:10]
                    ]
                }

                # Add job details
                for key, value in job_details.items():
                    if key not in result:  # Avoid overwriting
                        result[key] = value

                results.append(result)

            # Clean up temp file
            os.unlink(temp_file.name)

            return jsonify({
                'success': True,
                'matches': results
            })

        except Exception as e:
            # Clean up temp file
            os.unlink(temp_file.name)
            return jsonify({'error': str(e)}), 500
    else:
        return jsonify({'error': 'Invalid file type. Please upload a PDF.'}), 400


@app.route('/api/system_info', methods=['GET'])
def api_system_info():
    """API endpoint for system information"""
    system_info = job_system.get_system_info()
    return jsonify(system_info)


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files"""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


# Add a context processor to make configuration URL available in all templates
@app.context_processor
def inject_config_url():
    """Make configuration URL available in all templates"""
    return dict(config_url=url_for('config'))


if __name__ == '__main__':
    import pandas as pd
    import argparse

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Resume-Job Matching System')
    parser.add_argument('--metric', type=str,
                        choices=['cosine', 'euclidean', 'manhattan', 'chebyshev', 'minkowski', 'jaccard', 'hamming',
                                 'canberra'],
                        default='cosine', help='Distance metric to use for matching')
    parser.add_argument('--rebuild', action='store_true',
                        help='Force rebuild of the model')
    parser.add_argument('--port', type=int, default=5000,
                        help='Port number for the Flask server')

    args = parser.parse_args()

    # Update model parameters if specified on command line
    if args.metric:
        job_system.knn_matcher.metric = args.metric
        print(f"Using distance metric: {args.metric}")

    # Rebuild if requested or if metric changed
    if args.rebuild:
        print("Forcing model rebuild...")
        job_system.setup(force_rebuild=True, verbose=True)

    app.run(debug=True, port=args.port)