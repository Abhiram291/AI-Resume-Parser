"""
Resume Parser - Core Parsing Logic
Uses NLP and NER for extracting structured information from resumes
"""

import re
import spacy
from pathlib import Path
import PyPDF2
import pdfplumber
from docx import Document
from utils import setup_logger
from datetime import datetime

logger = setup_logger(__name__)


class ResumeParser:
    """
    Advanced Resume Parser using NLP techniques
    """
    
    def __init__(self):
        """Initialize parser with NLP models and patterns"""
        try:
            # Load spaCy model
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("spaCy model not found. Installing...")
            import os
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")
        
        # Define regex patterns
        self.patterns = {
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'phone': r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',
            'linkedin': r'(?:https?://)?(?:www\.)?linkedin\.com/in/[\w-]+/?',
            'github': r'(?:https?://)?(?:www\.)?github\.com/[\w-]+/?',
            'url': r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&/=]*)'
        }
        
        # Common skill keywords (expandable)
        self.skills_database = self._load_skills_database()
        
        # Education keywords
        self.education_keywords = [
            'bachelor', 'master', 'phd', 'b.tech', 'm.tech', 'b.s', 'm.s',
            'mba', 'degree', 'diploma', 'certification', 'university', 'college',
            'institute', 'school', 'graduated', 'graduation'
        ]
        
        # Experience keywords
        self.experience_keywords = [
            'experience', 'work', 'employment', 'position', 'role', 'job',
            'intern', 'internship', 'engineer', 'developer', 'manager', 'analyst',
            'consultant', 'specialist', 'lead', 'senior', 'junior'
        ]
    
    def _load_skills_database(self):
        """Load comprehensive skills database"""
        return {
            'programming_languages': [
                'python', 'java', 'javascript', 'c++', 'c#', 'ruby', 'php',
                'swift', 'kotlin', 'go', 'rust', 'scala', 'r', 'matlab',
                'typescript', 'perl', 'shell', 'bash'
            ],
            'ml_ai': [
                'machine learning', 'deep learning', 'nlp', 'computer vision',
                'tensorflow', 'pytorch', 'keras', 'scikit-learn', 'transformers',
                'bert', 'gpt', 'llm', 'neural networks', 'cnn', 'rnn', 'lstm'
            ],
            'web_frameworks': [
                'react', 'angular', 'vue', 'django', 'flask', 'fastapi',
                'node.js', 'express', 'spring', 'laravel', 'ruby on rails'
            ],
            'databases': [
                'sql', 'mysql', 'postgresql', 'mongodb', 'redis', 'cassandra',
                'oracle', 'dynamodb', 'elasticsearch', 'neo4j'
            ],
            'cloud_devops': [
                'aws', 'azure', 'gcp', 'docker', 'kubernetes', 'jenkins',
                'terraform', 'ansible', 'ci/cd', 'git', 'github', 'gitlab'
            ],
            'data_science': [
                'pandas', 'numpy', 'matplotlib', 'seaborn', 'jupyter',
                'data analysis', 'data visualization', 'statistics', 'tableau',
                'power bi', 'spark', 'hadoop'
            ],
            'soft_skills': [
                'leadership', 'communication', 'teamwork', 'problem solving',
                'analytical', 'critical thinking', 'agile', 'scrum'
            ]
        }
    
    def parse(self, file_path):
        """
        Main parsing method
        
        Args:
            file_path: Path to resume file
            
        Returns:
            dict: Structured resume data
        """
        logger.info(f"Starting parse for: {file_path}")
        
        # Extract text
        text = self._extract_text(file_path)
        
        # Process with NLP
        doc = self.nlp(text)
        
        # Extract all components
        resume_data = {
            'personal_info': self._extract_personal_info(text, doc),
            'education': self._extract_education(text, doc),
            'experience': self._extract_experience(text, doc),
            'skills': self._extract_skills(text),
            'certifications': self._extract_certifications(text, doc),
            'projects': self._extract_projects(text, doc),
            'languages': self._extract_languages(doc)
        }
        
        return resume_data
    
    def _extract_text(self, file_path):
        """Extract text from different file formats"""
        file_path = Path(file_path)
        extension = file_path.suffix.lower()
        
        try:
            if extension == '.pdf':
                return self._extract_from_pdf(file_path)
            elif extension == '.docx':
                return self._extract_from_docx(file_path)
            elif extension == '.txt':
                return self._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {extension}")
        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_path):
        """Extract text from PDF"""
        text = ""
        try:
            # Try pdfplumber first (better for complex layouts)
            with pdfplumber.open(file_path) as pdf:
                for page in pdf.pages:
                    text += page.extract_text() or ""
        except:
            # Fallback to PyPDF2
            with open(file_path, 'rb') as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    text += page.extract_text() or ""
        return text
    
    def _extract_from_docx(self, file_path):
        """Extract text from DOCX"""
        doc = Document(file_path)
        return "\n".join([para.text for para in doc.paragraphs])
    
    def _extract_from_txt(self, file_path):
        """Extract text from TXT"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    
    def _extract_personal_info(self, text, doc):
        """Extract personal information using NER and regex"""
        info = {}
        
        # Extract email
        email_match = re.search(self.patterns['email'], text)
        info['email'] = email_match.group(0) if email_match else None
        
        # Extract phone
        phone_match = re.search(self.patterns['phone'], text)
        info['phone'] = phone_match.group(0) if phone_match else None
        
        # Extract LinkedIn
        linkedin_match = re.search(self.patterns['linkedin'], text)
        info['linkedin'] = linkedin_match.group(0) if linkedin_match else None
        
        # Extract GitHub
        github_match = re.search(self.patterns['github'], text)
        info['github'] = github_match.group(0) if github_match else None
        
        # Extract name - improved logic
        # First, try to get name from first few lines (usually at top of resume)
        lines = text.split('\n')
        name = None
        
        # Check first 5 lines for a name (2-4 words, capitalized, not too long)
        for line in lines[:5]:
            line = line.strip()
            # Clean special characters and encoding issues
            line = re.sub(r'[^\w\s]', '', line)
            words = line.split()
            
            # Name is typically 2-4 words, each capitalized, reasonable length
            if 2 <= len(words) <= 4 and all(w[0].isupper() for w in words if w) and len(line) < 50:
                # Avoid lines with common resume keywords
                keywords = ['university', 'college', 'bachelor', 'master', 'degree', 
                           'email', 'phone', 'linkedin', 'github', 'india', 'usa']
                if not any(kw in line.lower() for kw in keywords):
                    name = line
                    break
        
        # Fallback to NER if above method fails
        if not name:
            persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
            # Filter out obvious non-names
            persons = [p for p in persons if len(p.split()) <= 4 and len(p) < 50]
            name = persons[0] if persons else None
        
        info['name'] = name
        
        # Extract location
        locations = [ent.text for ent in doc.ents if ent.label_ in ['GPE', 'LOC']]
        info['location'] = locations[0] if locations else None
        
        return info
    
    def _extract_education(self, text, doc):
        """Extract education information"""
        education = []
        lines = text.split('\n')
        
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Check if line contains education keywords
            if any(keyword in line_lower for keyword in self.education_keywords):
                edu_entry = {}
                
                # Extract degree
                degree_patterns = [
                    r'(bachelor|master|phd|b\.tech|m\.tech|b\.s|m\.s|mba).*?(?:in|of)?\s+([a-z\s]+)',
                    r'(degree|diploma)\s+in\s+([a-z\s]+)'
                ]
                for pattern in degree_patterns:
                    match = re.search(pattern, line_lower)
                    if match:
                        edu_entry['degree'] = line.strip()
                        break
                
                # Extract institution (usually in same or next line)
                orgs = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
                if orgs:
                    edu_entry['institution'] = orgs[0]
                
                # Extract year
                year_match = re.search(r'\b(19|20)\d{2}\b', line)
                if year_match:
                    edu_entry['year'] = year_match.group(0)
                
                # Extract GPA
                gpa_match = re.search(r'gpa[:\s]+(\d+\.?\d*)', line_lower)
                if gpa_match:
                    edu_entry['GPA'] = gpa_match.group(1)
                
                if edu_entry:
                    education.append(edu_entry)
        
        return education if education else []
    
    def _extract_experience(self, text, doc):
        """Extract work experience"""
        experience = []
        lines = text.split('\n')
        
        current_exp = {}
        for line in lines:
            line_lower = line.lower()
            
            # Check for company/organization
            orgs = [ent.text for ent in self.nlp(line).ents if ent.label_ == 'ORG']
            if orgs and any(keyword in line_lower for keyword in self.experience_keywords):
                if current_exp:
                    experience.append(current_exp)
                
                current_exp = {
                    'company': orgs[0],
                    'role': line.strip(),
                    'duration': None,
                    'responsibilities': []
                }
                
                # Extract duration
                date_pattern = r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}\s*-\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4}|present'
                date_match = re.search(date_pattern, line_lower)
                if date_match:
                    current_exp['duration'] = date_match.group(0)
            
            # Collect responsibilities (bullet points or dashed lines)
            elif current_exp and (line.strip().startswith(('•', '-', '●', '◦')) or 
                                  re.match(r'^\s*[\*\-]\s', line)):
                responsibility = re.sub(r'^[\s•\-●◦\*]+', '', line).strip()
                if responsibility:
                    current_exp['responsibilities'].append(responsibility)
        
        if current_exp:
            experience.append(current_exp)
        
        return experience
    
    def _extract_skills(self, text):
        """Extract skills using keyword matching"""
        text_lower = text.lower()
        extracted_skills = set()
        
        # Search for skills from database
        for category, skills in self.skills_database.items():
            for skill in skills:
                # Use word boundaries for exact matching
                pattern = r'\b' + re.escape(skill.lower()) + r'\b'
                if re.search(pattern, text_lower):
                    extracted_skills.add(skill.title())
        
        return sorted(list(extracted_skills))
    
    def _extract_certifications(self, text, doc):
        """Extract certifications"""
        certifications = []
        lines = text.split('\n')
        
        cert_keywords = ['certification', 'certified', 'certificate', 'license']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in cert_keywords):
                cert = line.strip()
                if cert:
                    certifications.append(cert)
        
        return certifications
    
    def _extract_projects(self, text, doc):
        """Extract projects"""
        projects = []
        lines = text.split('\n')
        
        project_keywords = ['project', 'developed', 'built', 'created', 'implemented']
        
        for line in lines:
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in project_keywords):
                project = line.strip()
                if project and len(project) > 20:  # Filter out very short matches
                    projects.append(project)
        
        return projects[:10]  # Limit to top 10
    
    def _extract_languages(self, doc):
        """Extract spoken/written languages"""
        languages = []
        
        # Common languages
        common_languages = [
            'english', 'spanish', 'french', 'german', 'chinese', 'japanese',
            'korean', 'arabic', 'hindi', 'portuguese', 'russian', 'italian'
        ]
        
        text_lower = doc.text.lower()
        for lang in common_languages:
            if lang in text_lower:
                languages.append(lang.title())
        
        return list(set(languages))