# test_parser.py

import os
import argparse
from resume_parser import ResumeParser


def test_parser_on_file(pdf_path, output_file=None):
    """Test the parser on a single file"""
    parser = ResumeParser(ocr_enabled=True)
    result = parser.parse(pdf_path)

    print(f"File: {result.get('filename')}")
    print(f"Extraction method: {result.get('extraction_method')}")
    print(f"Character count: {result.get('char_count')}")
    print(f"Word count: {result.get('word_count')}")
    print("\nFirst 500 characters:")
    print(result.get('text', '')[:500])

    if output_file:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(result.get('text', ''))
        print(f"\nFull text saved to {output_file}")


def test_parser_on_directory(directory, output_dir=None):
    """Test the parser on all PDFs in a directory"""
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    parser = ResumeParser(ocr_enabled=True)

    for filename in os.listdir(directory):
        if filename.lower().endswith('.pdf'):
            file_path = os.path.join(directory, filename)
            print(f"\nProcessing {filename}...")

            result = parser.parse(file_path)

            print(f"Extraction method: {result.get('extraction_method')}")
            print(f"Character count: {result.get('char_count')}")
            print(f"Word count: {result.get('word_count')}")

            if output_dir:
                output_file = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.txt")
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(result.get('text', ''))
                print(f"Text saved to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test resume PDF parser')
    parser.add_argument('path', help='Path to PDF file or directory of PDF files')
    parser.add_argument('--output', '-o', help='Output file or directory for extracted text')

    args = parser.parse_args()

    if os.path.isdir(args.path):
        test_parser_on_directory(args.path, args.output)
    else:
        test_parser_on_file(args.path, args.output)