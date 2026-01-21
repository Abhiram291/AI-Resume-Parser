"""
Resume Parser - Main Entry Point
Advanced NLP-based Resume Parser with Skill Matching
"""

import argparse
import json
import os
from pathlib import Path
from resume_parser import ResumeParser
from skill_matcher import SkillMatcher
from utils import setup_logger, save_json, load_json

logger = setup_logger(__name__)


def parse_resume(resume_path, output_dir="output"):
    """
    Parse a single resume and extract structured information
    
    Args:
        resume_path: Path to resume file (PDF, DOCX, or TXT)
        output_dir: Directory to save output JSON
    
    Returns:
        dict: Parsed resume data
    """
    try:
        logger.info(f"Parsing resume: {resume_path}")
        
        # Initialize parser
        parser = ResumeParser()
        
        # Parse resume
        resume_data = parser.parse(resume_path)
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, 
            f"{Path(resume_path).stem}_parsed.json"
        )
        save_json(resume_data, output_file)
        
        logger.info(f"Resume parsed successfully. Output saved to: {output_file}")
        return resume_data
        
    except Exception as e:
        logger.error(f"Error parsing resume: {str(e)}")
        raise


def match_with_job(resume_path, job_description_path, output_dir="output"):
    """
    Parse resume and match with job description
    
    Args:
        resume_path: Path to resume file
        job_description_path: Path to job description JSON or text file
        output_dir: Directory to save output
    
    Returns:
        dict: Parsed resume with skill matching results
    """
    try:
        logger.info(f"Matching resume with job description...")
        
        # Parse resume
        parser = ResumeParser()
        resume_data = parser.parse(resume_path)
        
        # Load job description
        if job_description_path.endswith('.json'):
            job_desc = load_json(job_description_path)
        else:
            with open(job_description_path, 'r', encoding='utf-8') as f:
                job_desc = {"description": f.read()}
        
        # Match skills
        matcher = SkillMatcher()
        match_results = matcher.match(resume_data, job_desc)
        
        # Combine results
        resume_data['skill_match'] = match_results
        
        # Save output
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(
            output_dir, 
            f"{Path(resume_path).stem}_matched.json"
        )
        save_json(resume_data, output_file)
        
        logger.info(f"Skill matching completed. Output saved to: {output_file}")
        return resume_data
        
    except Exception as e:
        logger.error(f"Error in skill matching: {str(e)}")
        raise


def batch_process(resume_dir, job_description_path=None, output_dir="output"):
    """
    Process multiple resumes in batch
    
    Args:
        resume_dir: Directory containing resumes
        job_description_path: Optional job description for matching
        output_dir: Directory to save outputs
    """
    resume_files = []
    for ext in ['*.pdf', '*.docx', '*.txt']:
        resume_files.extend(Path(resume_dir).glob(ext))
    
    logger.info(f"Found {len(resume_files)} resumes to process")
    
    results = []
    for resume_file in resume_files:
        try:
            if job_description_path:
                result = match_with_job(str(resume_file), job_description_path, output_dir)
            else:
                result = parse_resume(str(resume_file), output_dir)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to process {resume_file}: {str(e)}")
    
    # Save batch summary
    summary_file = os.path.join(output_dir, "batch_summary.json")
    save_json(results, summary_file)
    logger.info(f"Batch processing completed. Summary saved to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="AI-Powered Resume Parser with Skill Matching"
    )
    
    parser.add_argument(
        '--resume',
        type=str,
        help='Path to resume file (PDF, DOCX, or TXT)'
    )
    
    parser.add_argument(
        '--job',
        type=str,
        help='Path to job description (JSON or TXT)'
    )
    
    parser.add_argument(
        '--batch',
        type=str,
        help='Directory containing multiple resumes for batch processing'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='output',
        help='Output directory (default: output)'
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if not (args.resume or args.batch):
        parser.error("Either --resume or --batch must be specified")
    
    # Process
    if args.batch:
        batch_process(args.batch, args.job, args.output)
    elif args.job:
        match_with_job(args.resume, args.job, args.output)
    else:
        parse_resume(args.resume, args.output)
    
    logger.info("Processing complete!")


if __name__ == "__main__":
    main()