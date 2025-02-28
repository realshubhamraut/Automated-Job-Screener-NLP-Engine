import re
import os
import spacy
from typing import Dict, List, Any, Optional, Set, Tuple
import logging
from collections import defaultdict

from src.utils.logger import get_logger

logger = get_logger(__name__)

class EntityExtractor:
    """
    Advanced entity extraction for resumes and job descriptions.
    Extracts skills, experience, education, and other relevant entities.
    """
    
    def __init__(self, model_name: str = "en_core_web_md", use_gpu: bool = False):
        """
        Initialize the entity extractor with a spaCy model
        
        Args:
            model_name (str): Name of the spaCy model to use
            use_gpu (bool): Whether to use GPU acceleration
        """
        self.model_name = model_name
        
        # Try to load the spaCy model
        try:
            self.nlp = spacy.load(model_name)
            logger.info(f"Loaded spaCy model: {model_name}")
        except OSError:
            # If the model is not found, download it
            logger.info(f"Downloading spaCy model: {model_name}")
            spacy.cli.download(model_name)
            self.nlp = spacy.load(model_name)
        
        # Configure GPU if requested and available
        if use_gpu:
            try:
                spacy.require_gpu()
                logger.info("Using GPU for spaCy")
            except Exception as e:
                logger.warning(f"Failed to enable GPU for spaCy: {str(e)}")
        
        # Load skill patterns from files if available
        self.skill_patterns = self._load_skill_patterns()
        
        # Compile regex patterns for entity extraction
        self._compile_regex_patterns()
    
    def _load_skill_patterns(self) -> Set[str]:
        """
        Load skill patterns from a file
        
        Returns:
            Set[str]: Set of skill patterns
        """
        skills = set()
        
        # Try to load from common paths
        possible_paths = [
            os.path.join(os.path.dirname(__file__), '../../data/skills.txt'),
            os.path.join(os.path.dirname(__file__), '../data/skills.txt'),
            os.path.join(os.path.dirname(__file__), 'data/skills.txt'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r', encoding='utf-8') as f:
                        skills.update([line.strip().lower() for line in f if line.strip()])
                    logger.info(f"Loaded {len(skills)} skills from {path}")
                    break
                except Exception as e:
                    logger.error(f"Error loading skills from {path}: {str(e)}")
        
        # Default skills if no file is found
        if not skills:
            logger.warning("No skill file found. Using default skill list.")
            default_skills = {
                "python", "java", "javascript", "c++", "c#", "sql", "react", "node.js", 
                "machine learning", "data science", "artificial intelligence", "aws", 
                "docker", "kubernetes", "terraform", "git", "agile", "scrum", "project management"
            }
            skills.update(default_skills)
        
        return skills
    
    def _compile_regex_patterns(self):
        """Compile regex patterns for entity extraction"""
        
        # Email pattern
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        
        # Phone number pattern (various formats)
        self.phone_pattern = re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b')
        
        # URL pattern
        self.url_pattern = re.compile(r'https?://(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)')
        
        # Education pattern (degrees)
        self.education_pattern = re.compile(r'\b(?:Bachelor|Master|PhD|B\.S\.|M\.S\.|M\.B\.A\.|B\.A\.|B\.E\.|M\.E\.|B\.Tech|M\.Tech|Ph\.D)\b')
        
        # Experience pattern (years)
        self.experience_pattern = re.compile(r'\b(\d+)[\+]?\s*(?:years?|yrs?)(?:\s+of)?\s+(?:experience|exp)(?:rience)?\b', re.IGNORECASE)
        
        # Date pattern
        self.date_pattern = re.compile(r'\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)[,.\s]+\d{4}\b|\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b')
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities from text
        
        Args:
            text (str): Text to extract entities from
            
        Returns:
            Dict[str, List[str]]: Dictionary of entity types and their values
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for entity extraction")
            return {}
        
        # Initialize entity dictionary
        entities = defaultdict(list)
        
        # Process the text with spaCy
        doc = self.nlp(text)
        
        # Extract standard named entities
        for ent in doc.ents:
            # Skip very short entities (likely noise)
            if len(ent.text.strip()) < 2:
                continue
                
            # Add entity to the appropriate type
            entities[ent.label_].append(ent.text.strip())
        
        # Extract skills using pattern matching
        extracted_skills = self.extract_skills(text)
        if extracted_skills:
            entities["SKILL"] = extracted_skills
        
        # Extract contact information
        emails = self.extract_emails(text)
        if emails:
            entities["EMAIL"] = emails
            
        phones = self.extract_phones(text)
        if phones:
            entities["PHONE"] = phones
            
        urls = self.extract_urls(text)
        if urls:
            entities["URL"] = urls
        
        # Extract education
        education = self.extract_education(text)
        if education:
            entities["EDUCATION"] = education
        
        # Extract experience
        experience = self.extract_experience(text)
        if experience:
            entities["EXPERIENCE"] = experience
        
        # Extract dates
        dates = self.extract_dates(text)
        if dates:
            entities["DATE"] = dates
        
        # Convert defaultdict to regular dict and remove duplicates
        result = {k: list(set(v)) for k, v in entities.items()}
        
        return result
    
    def extract_skills(self, text: str) -> List[str]:
        """
        Extract skills from text using pattern matching
        
        Args:
            text (str): Text to extract skills from
            
        Returns:
            List[str]: List of extracted skills
        """
        found_skills = []
        
        # Convert text to lowercase for case-insensitive matching
        text_lower = text.lower()
        
        # Match skills from our pattern list
        for skill in self.skill_patterns:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(skill) + r'\b'
            if re.search(pattern, text_lower):
                # Use the skill with proper capitalization from original text
                # Find the actual occurrence in the text
                match = re.search(pattern, text_lower)
                if match:
                    start, end = match.span()
                    original_case = text[start:end]
                    found_skills.append(original_case)
        
        return found_skills
    
    def extract_emails(self, text: str) -> List[str]:
        """
        Extract email addresses from text
        
        Args:
            text (str): Text to extract emails from
            
        Returns:
            List[str]: List of extracted emails
        """
        return list(set(self.email_pattern.findall(text)))
    
    def extract_phones(self, text: str) -> List[str]:
        """
        Extract phone numbers from text
        
        Args:
            text (str): Text to extract phone numbers from
            
        Returns:
            List[str]: List of extracted phone numbers
        """
        return list(set(self.phone_pattern.findall(text)))
    
    def extract_urls(self, text: str) -> List[str]:
        """
        Extract URLs from text
        
        Args:
            text (str): Text to extract URLs from
            
        Returns:
            List[str]: List of extracted URLs
        """
        return list(set(self.url_pattern.findall(text)))
    
    def extract_education(self, text: str) -> List[str]:
        """
        Extract education information from text
        
        Args:
            text (str): Text to extract education from
            
        Returns:
            List[str]: List of extracted education information
        """
        # Find paragraphs that contain education keywords
        education_sections = []
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            if re.search(r'\b(?:education|degree|university|college|school)\b', para, re.IGNORECASE):
                # Extract sentences containing degree information
                degree_matches = self.education_pattern.findall(para)
                if degree_matches:
                    education_sections.append(para.strip())
                    continue
                    
                # Look for university/college names
                doc = self.nlp(para)
                if any(ent.label_ == "ORG" for ent in doc.ents):
                    education_sections.append(para.strip())
        
        return education_sections
    
    def extract_experience(self, text: str) -> List[str]:
        """
        Extract work experience information from text
        
        Args:
            text (str): Text to extract experience from
            
        Returns:
            List[str]: List of extracted experience information
        """
        # Find paragraphs that contain work experience keywords
        experience_sections = []
        paragraphs = re.split(r'\n\s*\n', text)
        
        for para in paragraphs:
            if re.search(r'\b(?:experience|work|employment|job|position)\b', para, re.IGNORECASE):
                # Extract sentences containing company/organization
                doc = self.nlp(para)
                if any(ent.label_ == "ORG" for ent in doc.ents) or self.experience_pattern.search(para):
                    experience_sections.append(para.strip())
        
        return experience_sections
    
    def extract_dates(self, text: str) -> List[str]:
        """
        Extract dates from text
        
        Args:
            text (str): Text to extract dates from
            
        Returns:
            List[str]: List of extracted dates
        """
        return list(set(self.date_pattern.findall(text)))
    
    def process_document(self, document: Dict[str, Any], text_field: str = 'original_text') -> Dict[str, Any]:
        """
        Process a document and add extracted entities to it
        
        Args:
            document (Dict[str, Any]): Document to process
            text_field (str): Field containing the text to process
            
        Returns:
            Dict[str, Any]: Document with added entities
        """
        if text_field not in document:
            logger.warning(f"Text field '{text_field}' not found in document")
            return document
        
        try:
            # Extract entities from the document text
            entities = self.extract_entities(document[text_field])
            
            # Add entities to the document
            if 'processed' not in document:
                document['processed'] = {}
                
            document['processed']['entities'] = entities
            
        except Exception as e:
            logger.error(f"Error processing document for entity extraction: {str(e)}")
        
        return document
    
    def extract_job_requirements(self, text: str) -> Dict[str, List[str]]:
        """
        Extract job requirements from job description text
        
        Args:
            text (str): Job description text
            
        Returns:
            Dict[str, List[str]]: Dictionary of requirement types and their values
        """
        requirements = defaultdict(list)
        
        # Look for specific sections in the text
        sections = {
            'required_skills': [r'required skills', r'technical skills', r'qualifications'],
            'experience': [r'experience', r'background'],
            'education': [r'education', r'academic', r'degree'],
            'responsibilities': [r'responsibilities', r'duties', r'what you\'ll do'],
        }
        
        for section_type, patterns in sections.items():
            for pattern in patterns:
                # Look for the section headers and extract content
                regex = re.compile(f'(?:{pattern})(?::|.|\n)+(.*?)(?:\n\s*\n|$)', re.IGNORECASE)
                matches = regex.findall(text)
                for match in matches:
                    if len(match.strip()) > 10:  # Avoid tiny matches
                        requirements[section_type].append(match.strip())
        
        # Extract skills specifically
        requirements['skills'] = self.extract_skills(text)
        
        # Convert defaultdict to regular dict
        return {k: list(set(v)) for k, v in requirements.items() if v}
    
    def extract_resume_sections(self, text: str) -> Dict[str, str]:
        """
        Extract standard resume sections
        
        Args:
            text (str): Resume text
            
        Returns:
            Dict[str, str]: Dictionary of section names and their content
        """
        sections = {}
        
        # Define section patterns to look for
        section_patterns = [
            ('summary', r'(?:summary|profile|objective)(?::|.|\n)+(.*?)(?:\n\s*\n|$)'),
            ('experience', r'(?:experience|employment|work history)(?::|.|\n)+(.*?)(?:\n\s*\n|$)'),
            ('education', r'(?:education|academic|qualifications)(?::|.|\n)+(.*?)(?:\n\s*\n|$)'),
            ('skills', r'(?:skills|expertise|technologies)(?::|.|\n)+(.*?)(?:\n\s*\n|$)'),
            ('projects', r'(?:projects|portfolio)(?::|.|\n)+(.*?)(?:\n\s*\n|$)'),
            ('certifications', r'(?:certifications|certificates|licenses)(?::|.|\n)+(.*?)(?:\n\s*\n|$)'),
        ]
        
        for section_name, pattern in section_patterns:
            regex = re.compile(pattern, re.IGNORECASE)
            matches = regex.findall(text)
            if matches:
                # Use the longest match as it's likely the most complete section
                best_match = max(matches, key=len)
                if len(best_match.strip()) > 10:  # Avoid tiny matches
                    sections[section_name] = best_match.strip()
        
        return sections