"""
Skill Matcher - Job Matching and Scoring Module
Uses sentence embeddings and cosine similarity for semantic matching
"""

import re
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from utils import setup_logger

logger = setup_logger(__name__)


class SkillMatcher:
    """
    Matches candidate skills with job requirements using NLP
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        """
        Initialize skill matcher with sentence transformer model
        
        Args:
            model_name: Hugging Face model name for embeddings
        """
        logger.info(f"Loading sentence transformer model: {model_name}")
        try:
            self.model = SentenceTransformer(model_name)
        except:
            logger.warning("Model not found. Downloading...")
            self.model = SentenceTransformer(model_name)
        
        # Common tech skills for better matching
        self.tech_keywords = [
            'python', 'java', 'javascript', 'react', 'node.js', 'aws', 'docker',
            'kubernetes', 'machine learning', 'deep learning', 'nlp', 'sql',
            'mongodb', 'tensorflow', 'pytorch', 'git', 'agile', 'scrum'
        ]
    
    def match(self, resume_data, job_description):
        """
        Match resume with job description and compute score
        
        Args:
            resume_data: Parsed resume dictionary
            job_description: Job description dict or string
            
        Returns:
            dict: Matching results with score and missing skills
        """
        logger.info("Starting skill matching...")
        
        # Extract job requirements
        if isinstance(job_description, dict):
            job_text = job_description.get('description', '')
            job_title = job_description.get('title', 'Unknown Position')
            required_skills = job_description.get('required_skills', [])
        else:
            job_text = job_description
            job_title = self._extract_job_title(job_text)
            required_skills = []
        
        # Extract skills from job description if not provided
        if not required_skills:
            required_skills = self._extract_skills_from_text(job_text)
        
        # Get candidate skills
        candidate_skills = resume_data.get('skills', [])
        
        # Compute matching score
        match_score = self._compute_match_score(
            candidate_skills, 
            required_skills,
            resume_data,
            job_text
        )
        
        # Find missing skills
        missing_skills = self._find_missing_skills(
            candidate_skills,
            required_skills
        )
        
        # Additional analysis
        matched_skills = self._find_matched_skills(
            candidate_skills,
            required_skills
        )
        
        results = {
            'job_title': job_title,
            'overall_score': round(match_score, 2),
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'total_required_skills': len(required_skills),
            'total_matched_skills': len(matched_skills),
            'match_percentage': round(
                (len(matched_skills) / len(required_skills) * 100) 
                if required_skills else 0, 
                2
            ),
            'recommendation': self._generate_recommendation(match_score)
        }
        
        logger.info(f"Match score: {match_score}%")
        return results
    
    def _extract_job_title(self, job_text):
        """Extract job title from text"""
        lines = job_text.split('\n')
        # Usually title is in first few lines
        for line in lines[:5]:
            if line.strip() and len(line.strip()) < 100:
                return line.strip()
        return "Unknown Position"
    
    def _extract_skills_from_text(self, text):
        """Extract skills from job description text"""
        text_lower = text.lower()
        extracted_skills = []
        
        # Use tech keywords
        for skill in self.tech_keywords:
            pattern = r'\b' + re.escape(skill.lower()) + r'\b'
            if re.search(pattern, text_lower):
                extracted_skills.append(skill.title())
        
        # Look for common patterns
        skill_patterns = [
            r'(?:experience with|knowledge of|proficient in|skilled in)\s+([^.,\n]+)',
            r'(?:required|must have|should have):\s*([^.\n]+)',
            r'skills?:\s*([^.\n]+)'
        ]
        
        for pattern in skill_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                skills_text = match.group(1)
                # Split by common delimiters
                skills = re.split(r'[,;/&]|\band\b', skills_text)
                extracted_skills.extend([s.strip().title() for s in skills if s.strip()])
        
        return list(set(extracted_skills))
    
    def _compute_match_score(self, candidate_skills, required_skills, resume_data, job_text):
        """
        Compute overall match score using multiple factors
        
        Args:
            candidate_skills: List of candidate's skills
            required_skills: List of required skills
            resume_data: Full resume data
            job_text: Job description text
            
        Returns:
            float: Match score (0-100)
        """
        if not required_skills:
            return 0.0
        
        # Factor 1: Exact skill matching (40% weight)
        exact_matches = len(set(
            [s.lower() for s in candidate_skills]
        ).intersection(
            [s.lower() for s in required_skills]
        ))
        exact_score = (exact_matches / len(required_skills)) * 40
        
        # Factor 2: Semantic similarity (40% weight)
        semantic_score = self._compute_semantic_similarity(
            candidate_skills,
            required_skills
        ) * 40
        
        # Factor 3: Experience relevance (20% weight)
        experience_score = self._compute_experience_score(
            resume_data,
            job_text
        ) * 20
        
        total_score = exact_score + semantic_score + experience_score
        return min(total_score, 100.0)
    
    def _compute_semantic_similarity(self, candidate_skills, required_skills):
        """Compute semantic similarity using sentence embeddings"""
        if not candidate_skills or not required_skills:
            return 0.0
        
        try:
            # Generate embeddings
            candidate_embeddings = self.model.encode(candidate_skills)
            required_embeddings = self.model.encode(required_skills)
            
            # Compute cosine similarity matrix
            similarity_matrix = cosine_similarity(
                candidate_embeddings,
                required_embeddings
            )
            
            # For each required skill, find best matching candidate skill
            max_similarities = similarity_matrix.max(axis=0)
            
            # Average similarity
            avg_similarity = max_similarities.mean()
            
            return float(avg_similarity)
            
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {str(e)}")
            return 0.0
    
    def _compute_experience_score(self, resume_data, job_text):
        """
        Compute score based on experience relevance
        
        Args:
            resume_data: Resume data dictionary
            job_text: Job description text
            
        Returns:
            float: Experience score (0-1)
        """
        try:
            # Extract experience descriptions
            experiences = resume_data.get('experience', [])
            if not experiences:
                return 0.5  # Neutral score if no experience
            
            # Combine all responsibilities
            experience_text = []
            for exp in experiences:
                responsibilities = exp.get('responsibilities', [])
                experience_text.extend(responsibilities)
            
            if not experience_text:
                return 0.5
            
            # Encode experience and job description
            exp_embedding = self.model.encode([' '.join(experience_text)])
            job_embedding = self.model.encode([job_text])
            
            # Compute similarity
            similarity = cosine_similarity(exp_embedding, job_embedding)[0][0]
            
            return float(similarity)
            
        except Exception as e:
            logger.error(f"Error computing experience score: {str(e)}")
            return 0.5
    
    def _find_matched_skills(self, candidate_skills, required_skills):
        """Find skills that match between candidate and requirements"""
        candidate_lower = {s.lower(): s for s in candidate_skills}
        required_lower = {s.lower() for s in required_skills}
        
        matched = []
        for req_skill_lower in required_lower:
            if req_skill_lower in candidate_lower:
                matched.append(candidate_lower[req_skill_lower])
        
        return matched
    
    def _find_missing_skills(self, candidate_skills, required_skills):
        """
        Find skills that are required but missing from candidate
        Uses semantic similarity for better matching
        """
        candidate_lower = set([s.lower() for s in candidate_skills])
        required_lower = {s.lower(): s for s in required_skills}
        
        missing = []
        
        for req_skill_lower, req_skill_original in required_lower.items():
            # Exact match check
            if req_skill_lower not in candidate_lower:
                # Check semantic similarity
                is_similar = False
                try:
                    req_embedding = self.model.encode([req_skill_original])
                    
                    for cand_skill in candidate_skills:
                        cand_embedding = self.model.encode([cand_skill])
                        similarity = cosine_similarity(req_embedding, cand_embedding)[0][0]
                        
                        # Consider similar if similarity > 0.8
                        if similarity > 0.8:
                            is_similar = True
                            break
                    
                    if not is_similar:
                        missing.append(req_skill_original)
                except:
                    # Fallback to exact match
                    missing.append(req_skill_original)
        
        return missing
    
    def _generate_recommendation(self, score):
        """Generate hiring recommendation based on score"""
        if score >= 80:
            return "Strong Match - Highly Recommended"
        elif score >= 60:
            return "Good Match - Recommended"
        elif score >= 40:
            return "Moderate Match - Consider with training"
        else:
            return "Weak Match - May not be suitable"
    
    def batch_rank_candidates(self, resumes_data_list, job_description):
        """
        Rank multiple candidates for a job
        
        Args:
            resumes_data_list: List of parsed resume dictionaries
            job_description: Job description dict or string
            
        Returns:
            list: Ranked candidates with scores
        """
        logger.info(f"Ranking {len(resumes_data_list)} candidates...")
        
        ranked_candidates = []
        
        for idx, resume_data in enumerate(resumes_data_list):
            match_result = self.match(resume_data, job_description)
            
            candidate_info = {
                'candidate_id': idx + 1,
                'name': resume_data.get('personal_info', {}).get('name', 'Unknown'),
                'email': resume_data.get('personal_info', {}).get('email'),
                'score': match_result['overall_score'],
                'match_details': match_result
            }
            
            ranked_candidates.append(candidate_info)
        
        # Sort by score (descending)
        ranked_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        logger.info("Ranking completed")
        return ranked_candidates