"""Pytest configuration and fixtures."""

import pytest


@pytest.fixture
def sample_resume_text():
    """Sample resume text for testing."""
    return """
John Doe
Software Engineer
john.doe@email.com | (555) 123-4567

SUMMARY
Experienced software engineer with 5+ years of experience in Python and JavaScript.

EXPERIENCE
Senior Software Engineer | Tech Company | 2020-Present
- Led development of microservices architecture
- Improved system performance by 40%
- Mentored junior developers

Software Engineer | Startup Inc | 2018-2020
- Built REST APIs using Python/FastAPI
- Implemented CI/CD pipelines
- Collaborated with cross-functional teams

EDUCATION
Bachelor of Science in Computer Science
University of Technology | 2018

SKILLS
Python, JavaScript, TypeScript, FastAPI, React, AWS, Docker, Kubernetes
"""


@pytest.fixture
def sample_job_description():
    """Sample job description for testing."""
    return """
Senior Software Engineer

We are looking for an experienced Software Engineer to join our team.

Requirements:
- 5+ years of experience in software development
- Strong proficiency in Python
- Experience with cloud services (AWS, GCP)
- Knowledge of containerization (Docker, Kubernetes)
- Excellent communication skills

Responsibilities:
- Design and implement scalable systems
- Lead technical projects
- Mentor junior team members
- Participate in code reviews
"""
