"""
Utility Functions for Resume Parser
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name, log_file='resume_parser.log', level=logging.INFO):
    """
    Setup logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Log file path
        level: Logging level
        
    Returns:
        Logger instance
    """
    # Create logs directory if it doesn't exist
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(log_dir / log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def save_json(data, file_path, indent=2):
    """
    Save data to JSON file
    
    Args:
        data: Data to save (dict or list)
        file_path: Output file path
        indent: JSON indentation
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=indent, ensure_ascii=False)
        return True
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error saving JSON to {file_path}: {str(e)}")
        return False


def load_json(file_path):
    """
    Load data from JSON file
    
    Args:
        file_path: Input file path
        
    Returns:
        dict or list: Loaded data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error loading JSON from {file_path}: {str(e)}")
        return None


def validate_resume_data(resume_data):
    """
    Validate parsed resume data structure
    
    Args:
        resume_data: Parsed resume dictionary
        
    Returns:
        tuple: (is_valid, error_messages)
    """
    required_fields = [
        'personal_info', 'education', 'experience', 
        'skills', 'certifications', 'projects', 'languages'
    ]
    
    errors = []
    
    # Check required fields
    for field in required_fields:
        if field not in resume_data:
            errors.append(f"Missing required field: {field}")
    
    # Validate personal_info
    if 'personal_info' in resume_data:
        personal_info = resume_data['personal_info']
        if not personal_info.get('email') and not personal_info.get('phone'):
            errors.append("At least email or phone should be present")
    
    return (len(errors) == 0, errors)


def clean_text(text):
    """
    Clean and normalize text
    
    Args:
        text: Input text
        
    Returns:
        str: Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    # Remove special characters (keep alphanumeric, spaces, and common punctuation)
    # text = re.sub(r'[^\w\s@.\-+():/,]', '', text)
    
    return text.strip()


def extract_years_of_experience(experience_list):
    """
    Calculate total years of experience from experience list
    
    Args:
        experience_list: List of experience dictionaries
        
    Returns:
        float: Total years of experience
    """
    from dateutil import parser
    import re
    
    total_months = 0
    
    for exp in experience_list:
        duration = exp.get('duration', '')
        if not duration:
            continue
        
        try:
            # Parse duration string (e.g., "Jan 2020 - Dec 2022")
            parts = duration.lower().split('-')
            if len(parts) == 2:
                start_str = parts[0].strip()
                end_str = parts[1].strip()
                
                # Handle "present" or "current"
                if 'present' in end_str or 'current' in end_str:
                    end_date = datetime.now()
                else:
                    end_date = parser.parse(end_str, fuzzy=True)
                
                start_date = parser.parse(start_str, fuzzy=True)
                
                # Calculate months
                months = (end_date.year - start_date.year) * 12
                months += end_date.month - start_date.month
                total_months += months
                
        except Exception:
            # Try to extract year range
            years = re.findall(r'\b(19|20)\d{2}\b', duration)
            if len(years) >= 2:
                total_months += (int(years[-1]) - int(years[0])) * 12
    
    return round(total_months / 12, 1)


def generate_summary_report(resume_data, match_results=None):
    """
    Generate a text summary report
    
    Args:
        resume_data: Parsed resume data
        match_results: Optional skill matching results
        
    Returns:
        str: Formatted summary report
    """
    report = []
    report.append("=" * 60)
    report.append("RESUME ANALYSIS REPORT")
    report.append("=" * 60)
    report.append("")
    
    # Personal Info
    personal_info = resume_data.get('personal_info', {})
    report.append("CANDIDATE INFORMATION")
    report.append("-" * 60)
    report.append(f"Name: {personal_info.get('name', 'N/A')}")
    report.append(f"Email: {personal_info.get('email', 'N/A')}")
    report.append(f"Phone: {personal_info.get('phone', 'N/A')}")
    report.append(f"Location: {personal_info.get('location', 'N/A')}")
    report.append(f"LinkedIn: {personal_info.get('linkedin', 'N/A')}")
    report.append(f"GitHub: {personal_info.get('github', 'N/A')}")
    report.append("")
    
    # Education
    education = resume_data.get('education', [])
    report.append("EDUCATION")
    report.append("-" * 60)
    if education:
        for edu in education:
            report.append(f"• {edu.get('degree', 'N/A')}")
            report.append(f"  {edu.get('institution', 'N/A')} - {edu.get('year', 'N/A')}")
            if edu.get('GPA'):
                report.append(f"  GPA: {edu.get('GPA')}")
    else:
        report.append("No education information found")
    report.append("")
    
    # Experience
    experience = resume_data.get('experience', [])
    report.append("WORK EXPERIENCE")
    report.append("-" * 60)
    if experience:
        years_exp = extract_years_of_experience(experience)
        report.append(f"Total Experience: ~{years_exp} years")
        report.append("")
        for exp in experience:
            report.append(f"• {exp.get('role', 'N/A')} at {exp.get('company', 'N/A')}")
            report.append(f"  Duration: {exp.get('duration', 'N/A')}")
    else:
        report.append("No work experience found")
    report.append("")
    
    # Skills
    skills = resume_data.get('skills', [])
    report.append("SKILLS")
    report.append("-" * 60)
    if skills:
        report.append(f"Total Skills: {len(skills)}")
        # Group skills in rows of 5
        for i in range(0, len(skills), 5):
            report.append(", ".join(skills[i:i+5]))
    else:
        report.append("No skills found")
    report.append("")
    
    # Match Results
    if match_results:
        report.append("JOB MATCHING RESULTS")
        report.append("=" * 60)
        report.append(f"Job Title: {match_results.get('job_title', 'N/A')}")
        report.append(f"Overall Match Score: {match_results.get('overall_score', 0)}%")
        report.append(f"Recommendation: {match_results.get('recommendation', 'N/A')}")
        report.append("")
        report.append(f"Matched Skills ({len(match_results.get('matched_skills', []))}): ")
        for skill in match_results.get('matched_skills', []):
            report.append(f"  ✓ {skill}")
        report.append("")
        report.append(f"Missing Skills ({len(match_results.get('missing_skills', []))}): ")
        for skill in match_results.get('missing_skills', []):
            report.append(f"  ✗ {skill}")
        report.append("")
    
    report.append("=" * 60)
    
    return "\n".join(report)


def export_to_csv(resumes_data_list, output_file='resumes_export.csv'):
    """
    Export parsed resumes to CSV
    
    Args:
        resumes_data_list: List of parsed resume dictionaries
        output_file: Output CSV file path
    """
    import csv
    
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as f:
            # Define fields
            fieldnames = [
                'name', 'email', 'phone', 'location', 'linkedin', 'github',
                'total_skills', 'skills', 'years_experience', 'education_count',
                'certifications_count', 'projects_count'
            ]
            
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for resume_data in resumes_data_list:
                personal_info = resume_data.get('personal_info', {})
                skills = resume_data.get('skills', [])
                experience = resume_data.get('experience', [])
                
                row = {
                    'name': personal_info.get('name', ''),
                    'email': personal_info.get('email', ''),
                    'phone': personal_info.get('phone', ''),
                    'location': personal_info.get('location', ''),
                    'linkedin': personal_info.get('linkedin', ''),
                    'github': personal_info.get('github', ''),
                    'total_skills': len(skills),
                    'skills': ', '.join(skills[:10]),  # First 10 skills
                    'years_experience': extract_years_of_experience(experience),
                    'education_count': len(resume_data.get('education', [])),
                    'certifications_count': len(resume_data.get('certifications', [])),
                    'projects_count': len(resume_data.get('projects', []))
                }
                
                writer.writerow(row)
        
        logger = logging.getLogger(__name__)
        logger.info(f"Exported {len(resumes_data_list)} resumes to {output_file}")
        return True
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"Error exporting to CSV: {str(e)}")
        return False