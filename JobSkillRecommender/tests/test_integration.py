# test_integration.py

from resume_parser import ResumeParser
import os


def test_with_sample_resumes():
    """Integration test with sample resumes"""
    # Create test directory if it doesn't exist
    test_dir = "parser/test_outputs"
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    # Create parser
    parser = ResumeParser(ocr_enabled=True)

    # Define sample resume paths (add your sample paths here)
    sample_paths = [
        "sample_resumes/resume1.pdf",
        "sample_resumes/resume2.pdf",
        "sample_resumes/resume3.pdf",
        "sample_resumes/resume4.pdf"
    ]

    # Process each sample
    for sample_path in sample_paths:
        if os.path.exists(sample_path):
            result = parser.parse(sample_path)

            # Print results
            print(f"\nSample: {os.path.basename(sample_path)}")
            print(f"Extraction method: {result.get('extraction_method')}")
            print(f"Character count: {result.get('char_count')}")
            print(f"Word count: {result.get('word_count')}")

            # Save extracted text
            output_file = os.path.join(test_dir, f"{os.path.splitext(os.path.basename(sample_path))[0]}.txt")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result.get('text', ''))
            print(f"Text saved to {output_file}")
        else:
            print(f"Sample file not found: {sample_path}")


if __name__ == "__main__":
    test_with_sample_resumes()