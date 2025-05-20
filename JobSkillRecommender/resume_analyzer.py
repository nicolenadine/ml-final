# resume_analyzer.py

import re
import numpy as np
from resume_parser import ResumeParser


class ResumeAnalyzer:
    """
    Analyzes resume text to extract structured information and create feature vectors.
    """

    def __init__(self, skills_dictionary=None, top_terms=None):
        """Initialize with optional skills dictionary and top terms from job analysis"""
        self.parser = ResumeParser(ocr_enabled=False)
        self.skills_dictionary = skills_dictionary or self._load_default_skills()
        self.top_terms = top_terms or []


    def _load_default_skills(self):
        """Load default skills dictionary if none provided"""
        # Could load from a CSV or build a comprehensive list
        return ["Python", "Java", "C++", "JavaScript", "SQL", "R", "HTML", "CSS",
                "AWS", "Azure", "GCP", "Docker", "Kubernetes", "TensorFlow", "PyTorch", ...]


    def parse_resume(self, file_path):
        """Parse resume file and return structured information"""
        # Extract text
        result = self.parser.parse(file_path)
        text = result.get("text", "")

        # Extract structured information
        structured_data = {
            "skills": self.extract_skills(text),
            "education": self.extract_education(text),
            "experience": self.extract_experience(text),
            "full_text": text
        }

        return structured_data


    def extract_skills(self, text):
        """Extract skills from resume text"""
        found_skills = []
        for skill in self.skills_dictionary:
            if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
                found_skills.append(skill)

        return found_skills


    def extract_education(self, text):
        """Extract education information"""
        # Implementation details...
        return []


    def extract_experience(self, text):
        """Extract work experience information"""
        # Implementation details...
        return []


    def create_feature_vector(self, text):
        """Create a feature vector for the resume based on top terms"""
        if not self.top_terms:
            raise ValueError("Top terms list is empty. Cannot create feature vector.")

        vector = np.zeros(len(self.top_terms))

        for i, term in enumerate(self.top_terms):
            # Count occurrences of the term
            pattern = r'\b' + re.escape(term.lower()) + r'\b'
            count = len(re.findall(pattern, text.lower()))
            vector[i] = count

        return vector