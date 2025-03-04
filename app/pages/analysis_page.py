import streamlit as st
import pandas as pd
import re
import numpy as np
from typing import Dict, List, Any, Tuple
import torch
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import PyPDF2
import io

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Initialize the model for text understanding
@st.cache_resource
def load_model():
    try:
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from a PDF file."""
    text = ""
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text() + "\n"
    except Exception as e:
        logger.error(f"Error extracting PDF text: {str(e)}")
    return text

class ResumeAnalyzer:
    """Improved class to extract information from resumes"""
    
    def __init__(self, resume_text, model=None):
        self.text = resume_text
        self.model = model
        
        # Direct pattern extraction - most reliable for contact info
        self.contact_info = self._extract_contact_info()
        
        # Extract sections for further analysis
        self.sections = self._extract_sections()
        
        # Process text into segments for semantic search
        self.segments = self._create_segments()
        self.segment_embeddings = None
        
        # Generate embeddings if model is available
        if self.model and self.segments:
            try:
                self.segment_embeddings = self.model.encode(self.segments, convert_to_tensor=True)
            except Exception as e:
                logger.error(f"Error computing embeddings: {str(e)}")
    
    def _extract_contact_info(self) -> Dict[str, str]:
        """Extract contact information directly using regex patterns"""
        contact_info = {
            "name": "Not found",
            "email": "Not found",
            "phone": "Not found",
            "location": "Not found"
        }
        
        # 1. Extract email (most reliable)
        email_match = re.search(r'[\w.+-]+@[\w.-]+\.\w+', self.text)
        if email_match:
            contact_info["email"] = email_match.group(0)
        
        # 2. Extract phone number
        phone_patterns = [
            r'\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b',  # Standard US format
            r'\(\d{3}\)\s?\d{3}[-.\s]?\d{4}',      # (123) 456-7890
            r'\+\d{1,2}\s?\d{3}[-.\s]?\d{3}[-.\s]?\d{4}'  # +1 123-456-7890
        ]
        
        for pattern in phone_patterns:
            phone_match = re.search(pattern, self.text)
            if phone_match:
                contact_info["phone"] = phone_match.group(0)
                break
        
        # 3. Extract name - use the first few lines
        first_lines = self.text.split('\n')[:10]
        
        # Approach 1: Look for standalone name pattern
        for line in first_lines:
            line = line.strip()
            if 5 < len(line) < 40 and re.match(r'^[A-Z][a-z]+(?:[\s-][A-Z][a-z]+)+$', line):
                contact_info["name"] = line
                break
        
        # Approach 2: Check for labeled name
        if contact_info["name"] == "Not found":
            for line in first_lines:
                name_match = re.search(r'Name\s*[:-]\s*([A-Z][a-z]+(?:[\s-][A-Z][a-z]+)+)', line, re.IGNORECASE)
                if name_match:
                    contact_info["name"] = name_match.group(1).strip()
                    break
        
        # Approach 3: Look at the very first non-empty line
        if contact_info["name"] == "Not found":
            for line in first_lines:
                if line.strip() and len(line.strip()) > 3:
                    words = line.strip().split()
                    if 2 <= len(words) <= 4:
                        # Check if it looks like a name (all words capitalized)
                        if all(word[0].isupper() for word in words):
                            contact_info["name"] = line.strip()
                            break
        
        # 4. Extract location
        # Look for city, state patterns
        location_patterns = [
            r'[A-Z][a-z]+,\s*[A-Z]{2}\s*\d{5}',  # City, ST ZIP
            r'[A-Z][a-z]+,\s*[A-Z]{2}',          # City, ST
            r'[A-Z][a-z]+,\s*[A-Za-z\s]+',       # City, State
        ]
        
        for pattern in location_patterns:
            location_match = re.search(pattern, self.text)
            if location_match:
                contact_info["location"] = location_match.group(0)
                break
        
        # Try labeled location if not found
        if contact_info["location"] == "Not found":
            location_match = re.search(r'(?:Location|Address|City)[:\s]+([A-Za-z0-9\s,.-]+)', self.text, re.IGNORECASE)
            if location_match:
                contact_info["location"] = location_match.group(1).strip()
        
        return contact_info

    def _extract_sections(self) -> Dict[str, str]:
        """Extract sections from resume text"""
        sections = {}
        lines = self.text.split('\n')
        
        # Define section patterns
        section_headers = {
            "summary": ["summary", "professional summary", "profile", "about me", "objective"],
            "experience": ["experience", "work history", "employment", "professional experience", "work experience"],
            "education": ["education", "academic", "qualifications"],
            "skills": ["skills", "technical skills", "expertise", "technologies", "competencies"],
            "projects": ["projects", "portfolio"]
        }
        
        # Find section boundaries
        section_boundaries = []
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Check if this line is a section header
            for section, keywords in section_headers.items():
                if any(line_lower == keyword or line_lower.startswith(f"{keyword}:") for keyword in keywords):
                    section_boundaries.append((i, section))
                    break
        
        # Extract section content
        for i, (line_idx, section_name) in enumerate(section_boundaries):
            start_idx = line_idx + 1  # Start from the line after header
            
            # Determine where section ends
            if i < len(section_boundaries) - 1:
                end_idx = section_boundaries[i+1][0]
            else:
                end_idx = len(lines)
            
            # Get section content
            section_content = '\n'.join(lines[start_idx:end_idx]).strip()
            if section_content:
                sections[section_name] = section_content
        
        return sections

    def _create_segments(self) -> List[str]:
        """Create meaningful segments from text for semantic search"""
        # Split text by paragraph first
        paragraphs = re.split(r'\n\s*\n', self.text)
        segments = []
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            lines = para.split('\n')
            current_segment = ""
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Detect if line seems like a new bullet or section
                if re.match(r'^[\•\-\*\>\◦\■]', line) or (line.isupper() and len(line) < 30):
                    # Save previous segment if exists
                    if current_segment:
                        segments.append(current_segment)
                    current_segment = line
                else:
                    # Add to current segment
                    if current_segment:
                        current_segment += " " + line
                    else:
                        current_segment = line
            
            # Add final segment
            if current_segment:
                segments.append(current_segment)
        
        # Filter segments
        filtered_segments = []
        for seg in segments:
            seg = re.sub(r'\s+', ' ', seg).strip()
            if seg and 5 < len(seg) < 1000:  # Reasonable size
                filtered_segments.append(seg)
        
        return filtered_segments

    def semantic_search(self, query: str, context: str = None, top_k: int = 1) -> List[str]:
        """Use semantic search to find relevant segments"""
        if not self.model or not self.segments:
            return []
        
        try:
            # Process specific context if provided
            if context:
                context_segments = [s.strip() for s in context.split('\n') if s.strip()]
                if context_segments:
                    context_embeddings = self.model.encode(context_segments, convert_to_tensor=True)
                    query_embedding = self.model.encode(query, convert_to_tensor=True)
                    cos_scores = util.cos_sim(query_embedding, context_embeddings)[0]
                    top_indices = torch.topk(cos_scores, min(top_k, len(cos_scores)))[1]
                    return [context_segments[idx] for idx in top_indices]
            
            # Use full document if no context or context search failed
            query_embedding = self.model.encode(query, convert_to_tensor=True)
            cos_scores = util.cos_sim(query_embedding, self.segment_embeddings)[0]
            top_indices = torch.topk(cos_scores, min(top_k, len(cos_scores)))[1]
            return [self.segments[idx] for idx in top_indices]
        except Exception as e:
            logger.error(f"Semantic search error: {str(e)}")
            return []

    def get_current_title(self) -> str:
        """Extract candidate's current job title"""
        # Try direct pattern matching first
        title_patterns = [
            r'((?:Senior|Junior|Lead|Principal)?\s*(?:Software|Data|Product|Project|Business|Marketing|Sales|UX|UI|Frontend|Backend|Full.?Stack|DevOps|Cloud|Network|Systems|Security)[\s\w]+(?:Engineer|Developer|Analyst|Scientist|Designer|Manager|Architect|Consultant|Specialist))',
            r'(?:Title|Position|Role)[:\s]+([A-Za-z\s\-\/&]+)'
        ]
        
        # Check the top part of the resume
        top_text = '\n'.join(self.text.split('\n')[:15])
        
        for pattern in title_patterns:
            title_match = re.search(pattern, top_text)
            if title_match:
                return title_match.group(1).strip()
        
        # Try semantic search
        summary_section = self.sections.get('summary', '')
        title_results = self.semantic_search("What is the candidate's current job title?", 
                                            context=summary_section if summary_section else None)
        
        if title_results:
            # Extract job title from result
            for result in title_results:
                for pattern in title_patterns:
                    title_match = re.search(pattern, result)
                    if title_match:
                        return title_match.group(1).strip()
                
                # If no pattern match but result is short, might be the title itself
                if len(result.split()) <= 5:
                    return result
        
        return "Not found"