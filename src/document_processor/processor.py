import re
import string
from typing import Dict, Any, List, Optional
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from src.utils.logger import get_logger

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

logger = get_logger(__name__)

class DocumentProcessor:
    """
    Process documents by cleaning, extracting metadata,
    and preparing them for embedding and matching.
    """
    
    def __init__(self):
        """Initialize the document processor"""
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
    
    def process_document(self, text: str, doc_type: str = "resume") -> Dict[str, Any]:
        """
        Process a document
        
        Args:
            text (str): Document text
            doc_type (str): Type of document ('resume' or 'job_description')
            
        Returns:
            Dict containing processed document data
        """
        if not text or not isinstance(text, str):
            return {"clean_text": "", "metadata": {}, "sections": {}}
        
        try:
            # Clean the text
            clean_text = self.clean_text(text)
            
            # Extract metadata
            metadata = self.extract_metadata(clean_text, doc_type)
            
            # Extract sections
            sections = self.extract_sections(clean_text, doc_type)
            
            # Extract keywords
            keywords = self.extract_keywords(clean_text)
            
            # Create processed document
            processed = {
                "clean_text": clean_text,
                "metadata": metadata,
                "sections": sections,
                "keywords": keywords
            }
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"clean_text": text, "metadata": {}, "sections": {}, "keywords": []}
    
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text (str): Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple newlines with single newline
        text = re.sub(r'\n+', '\n', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters (keep letters, numbers, and some punctuation)
        text = re.sub(r'[^\w\s\.\,\-\:\;\(\)\[\]\{\}\/]', '', text)
        
        return text.strip()
    
    def extract_metadata(self, text: str, doc_type: str) -> Dict[str, Any]:
        """
        Extract metadata from document
        
        Args:
            text (str): Document text
            doc_type (str): Type of document
            
        Returns:
            Dictionary of metadata
        """
        metadata = {}
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
        if emails:
            metadata["emails"] = emails
        
        # Extract phone numbers
        phones = re.findall(r'\b(?:\+\d{1,3}[- ]?)?\(?\d{3}\)?[- ]?\d{3}[- ]?\d{4}\b', text)
        if phones:
            metadata["phones"] = phones
        
        # Extract URLs
        urls = re.findall(r'https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+', text)
        if urls:
            metadata["urls"] = urls
        
        # Extract additional metadata based on document type
        if doc_type == "resume":
            # Try to extract name (often at the beginning)
            lines = text.strip().split('\n')
            if lines:
                potential_name = lines[0].strip()
                if len(potential_name.split()) <= 4 and len(potential_name) <= 40:
                    metadata["name"] = potential_name
            
            # Education keywords
            education_keywords = ['degree', 'university', 'college', 'bachelor', 'master', 'phd', 'education']
            for keyword in education_keywords:
                if keyword in text.lower():
                    metadata["has_education"] = True
                    break
            
            # Experience indicators
            exp_patterns = [r'\b(\d+)\+?\s*years?\s*(?:of)?\s*experience\b', 
                            r'\bexperience\s*:?\s*(\d+)\+?\s*years?\b']
            for pattern in exp_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    try:
                        metadata["years_experience"] = int(matches[0])
                        break
                    except ValueError:
                        pass
                        
        elif doc_type == "job_description":
            # Try to extract job title
            title_patterns = [r'job\s*title\s*:?\s*([^\n]+)', 
                              r'position\s*:?\s*([^\n]+)',
                              r'role\s*:?\s*([^\n]+)']
            for pattern in title_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    metadata["job_title"] = matches[0].strip()
                    break
            
            # Try to extract company name
            company_patterns = [r'company\s*:?\s*([^\n]+)', 
                                r'organization\s*:?\s*([^\n]+)',
                                r'employer\s*:?\s*([^\n]+)']
            for pattern in company_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    metadata["company"] = matches[0].strip()
                    break
            
            # Try to extract location
            location_patterns = [r'location\s*:?\s*([^\n]+)', 
                                 r'place\s*:?\s*([^\n]+)',
                                 r'city\s*:?\s*([^\n]+)']
            for pattern in location_patterns:
                matches = re.findall(pattern, text.lower())
                if matches:
                    metadata["location"] = matches[0].strip()
                    break
                    
        return metadata
    
    def extract_sections(self, text: str, doc_type: str) -> Dict[str, str]:
        """
        Extract sections from document
        
        Args:
            text (str): Document text
            doc_type (str): Type of document
            
        Returns:
            Dictionary of sections
        """
        sections = {}
        
        # Define section patterns based on document type
        if doc_type == "resume":
            section_patterns = {
                'summary': r'(?:summary|profile|objective|about\s*me)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'experience': r'(?:experience|work\s*history|employment)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'education': r'(?:education|academic|qualifications|degrees)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'skills': r'(?:skills|technical\s*skills|technologies|competencies)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'projects': r'(?:projects|portfolio|works)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'certifications': r'(?:certifications|certificates|accreditations)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
            }
        else:  # job_description
            section_patterns = {
                'summary': r'(?:summary|overview|about\s*the\s*role)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'requirements': r'(?:requirements|qualifications|what\s*you\s*need)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'responsibilities': r'(?:responsibilities|duties|what\s*you\'ll\s*do)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'benefits': r'(?:benefits|perks|what\s*we\s*offer|compensation)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
                'company': r'(?:about\s*us|company|organization|who\s*we\s*are)(?:\s*:|\s*\n)(.*?)(?:\n\s*\n|\n\s*[A-Z])',
            }
        
        # Extract sections using regex
        for section_name, pattern in section_patterns.items():
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                sections[section_name] = matches.group(1).strip()
        
        return sections
    
    def extract_keywords(self, text: str) -> List[str]:
        """
        Extract important keywords from text
        
        Args:
            text (str): Document text
            
        Returns:
            List of keywords
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and punctuation
        tokens = [w for w in tokens if w.lower() not in self.stop_words and w not in string.punctuation]
        
        # Lemmatize
        tokens = [self.lemmatizer.lemmatize(w) for w in tokens]
        
        # Count word frequencies
        word_freq = {}
        for word in tokens:
            if len(word) > 2:  # Only consider words with length > 2
                if word in word_freq:
                    word_freq[word] += 1
                else:
                    word_freq[word] = 1
        
        # Sort by frequency
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        
        # Return top 30 keywords
        return [word for word, freq in sorted_words[:30]]