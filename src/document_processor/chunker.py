import re
from typing import List, Dict, Optional, Union
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

class TextChunker:
    """
    Splits text documents into chunks for processing and embedding
    """
    
    def __init__(self, chunk_size: int = 300, chunk_overlap: int = 50):
        """
        Initialize the text chunker
        
        Args:
            chunk_size (int): Size of each chunk in characters
            chunk_overlap (int): Overlap between consecutive chunks in characters
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Common section headers in resumes and job descriptions
        self.section_patterns = {
            'summary': r'(?:summary|profile|objective|about me)',
            'experience': r'(?:experience|work history|employment|work experience|career|professional experience)',
            'education': r'(?:education|academic|qualifications|degrees|training)',
            'skills': r'(?:skills|technical skills|technologies|competencies|expertise|proficiencies)',
            'projects': r'(?:projects|portfolio|works)',
            'certifications': r'(?:certifications|certificates|licenses|accreditations)',
            'awards': r'(?:awards|honors|achievements|accomplishments)',
            'publications': r'(?:publications|research|papers|articles)',
            'languages': r'(?:languages|language proficiency)',
            'volunteer': r'(?:volunteer|community service)',
            'interests': r'(?:interests|hobbies|activities)',
            'references': r'(?:references|testimonials)',
            'contact': r'(?:contact|personal information|details)',
            'requirements': r'(?:requirements|qualifications|what we\'re looking for)',
            'responsibilities': r'(?:responsibilities|duties|job description|what you\'ll do|role overview)',
            'benefits': r'(?:benefits|perks|compensation|what we offer)'
        }
        
        # Compile section regex
        self.section_regex = self._compile_section_regex()
    
    def _compile_section_regex(self):
        """
        Compile regex for identifying document sections
        
        Returns:
            re.Pattern: Compiled regex pattern for sections
        """
        # Create pattern that matches any section header
        patterns = []
        for section, pattern in self.section_patterns.items():
            # Match section headers that are either uppercase, title case, or followed by a colon
            patterns.append(f'(?P<{section}>(?:^|\n)[ \t]*(?:{pattern})[ \t]*(?::|[^\na-z]{{0,10}}$|\n))')
        
        return re.compile('|'.join(patterns), re.IGNORECASE | re.MULTILINE)
    
    def chunk_text(self, text: str) -> List[str]:
        """
        Split text into overlapping chunks of specified size
        
        Args:
            text (str): Text to split into chunks
            
        Returns:
            List[str]: List of text chunks
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for chunking")
            return []
        
        # If text is smaller than chunk size, return as single chunk
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            # Calculate end position
            end = start + self.chunk_size
            
            # If we're not at the end of the text, try to find a good break point
            if end < len(text):
                # Try to break at paragraph
                paragraph_break = text.rfind('\n\n', start, end)
                if paragraph_break != -1 and paragraph_break > start + self.chunk_size // 2:
                    end = paragraph_break + 2  # Include the double newline
                else:
                    # Try to break at sentence
                    sentence_breaks = [text.rfind('. ', start, end), text.rfind('! ', start, end), 
                                     text.rfind('? ', start, end), text.rfind('.\n', start, end)]
                    best_break = max(sentence_breaks)
                    
                    if best_break != -1 and best_break > start + self.chunk_size // 3:
                        end = best_break + 2  # Include the sentence end marker and space
                    else:
                        # Try to break at word boundary
                        space = text.rfind(' ', start, end)
                        if space != -1 and space > start + self.chunk_size // 2:
                            end = space + 1  # Include the space
            
            # Add chunk to list
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # Move to next chunk with overlap
            start = max(start + 1, end - self.chunk_overlap)
        
        return chunks
    
    def chunk_by_section(self, text: str) -> Dict[str, str]:
        """
        Split text into sections based on common section headers
        
        Args:
            text (str): Text to split into sections
            
        Returns:
            Dict[str, str]: Dictionary mapping section names to their content
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for section chunking")
            return {}
        
        # Find all section headers
        sections = {}
        last_position = 0
        last_section = 'other'
        
        # Add text start as first position
        matches = list(self.section_regex.finditer(text))
        
        if not matches:
            # No sections found, return whole text as 'other'
            return {'other': text}
        
        # Process each section
        for i, match in enumerate(matches):
            # Get section name (first named group that matched)
            section_name = next((name for name, value in match.groupdict().items() if value), 'other')
            
            # Get section start position
            section_start = match.start()
            
            # If this isn't the first match, save the previous section content
            if last_position > 0:
                section_content = text[last_position:section_start].strip()
                if section_content:
                    sections[last_section] = section_content
            
            # Update tracking variables
            last_position = match.end()
            last_section = section_name
        
        # Add the last section
        if last_position < len(text):
            section_content = text[last_position:].strip()
            if section_content:
                sections[last_section] = section_content
        
        return sections
    
    def chunk_document(self, document: Dict, text_field: str = 'original_text') -> List[Dict]:
        """
        Create chunks from a document dictionary
        
        Args:
            document (Dict): Document to chunk
            text_field (str): Field containing the text to chunk
            
        Returns:
            List[Dict]: List of chunk dictionaries with metadata
        """
        # Get text from document
        if text_field not in document:
            logger.warning(f"Text field '{text_field}' not found in document")
            return []
        
        text = document[text_field]
        chunks = self.chunk_text(text)
        
        # Create chunk dictionaries with document metadata
        chunk_dicts = []
        for i, chunk_text in enumerate(chunks):
            chunk_dict = {
                'chunk_id': f"{document.get('id', 'doc')}_{i}",
                'document_id': document.get('id', 'unknown'),
                'chunk_index': i,
                'text': chunk_text,
                'document_type': document.get('document_type', 'unknown'),
                'filename': document.get('filename', 'unknown')
            }
            chunk_dicts.append(chunk_dict)
        
        return chunk_dicts
    
    def chunk_documents(self, documents: List[Dict], text_field: str = 'original_text') -> List[Dict]:
        """
        Create chunks from multiple document dictionaries
        
        Args:
            documents (List[Dict]): List of documents to chunk
            text_field (str): Field containing the text to chunk
            
        Returns:
            List[Dict]: List of chunk dictionaries with metadata
        """
        all_chunks = []
        for doc in documents:
            doc_chunks = self.chunk_document(doc, text_field)
            all_chunks.extend(doc_chunks)
        
        return all_chunks