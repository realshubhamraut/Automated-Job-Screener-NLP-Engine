import os
import io
import PyPDF2
import docx
from typing import Union, BinaryIO, Any
import streamlit as st

from src.utils.logger import get_logger

logger = get_logger(__name__)

class DocumentLoader:
    """
    Load and extract text from various document formats.
    Handles both file paths and Streamlit UploadedFile objects.
    """
    
    def __init__(self):
        """Initialize the document loader"""
        pass
    
    def load_document(self, file_obj: Union[str, io.BytesIO, Any]) -> str:
        """
        Extract text from a document file
        
        Args:
            file_obj: Can be a file path, BytesIO object, or Streamlit UploadedFile
            
        Returns:
            Extracted text
        """
        try:
            # Handle Streamlit UploadedFile
            if hasattr(file_obj, 'name') and hasattr(file_obj, 'read'):
                filename = file_obj.name
                file_content = file_obj.read()
                
                # Create BytesIO object
                file_stream = io.BytesIO(file_content)
                
                # Reset file obj position for potential reuse
                if hasattr(file_obj, 'seek'):
                    file_obj.seek(0)
                
            # Handle file paths
            elif isinstance(file_obj, str):
                filename = os.path.basename(file_obj)
                with open(file_obj, 'rb') as f:
                    file_content = f.read()
                file_stream = io.BytesIO(file_content)
            
            # Handle BytesIO objects
            elif isinstance(file_obj, io.BytesIO):
                filename = "unknown"
                file_stream = file_obj
            
            else:
                raise TypeError(f"Unsupported file object type: {type(file_obj)}")
            
            # Get file extension
            _, ext = os.path.splitext(filename.lower())
            
            # Extract text based on file type
            if ext == '.pdf':
                return self._extract_from_pdf(file_stream)
            elif ext == '.docx':
                return self._extract_from_docx(file_stream)
            elif ext == '.txt' or ext == '.md':
                # Reset stream position
                file_stream.seek(0)
                return file_stream.read().decode('utf-8', errors='replace')
            else:
                raise ValueError(f"Unsupported file format: {ext}")
                
        except Exception as e:
            logger.error(f"Error loading document: {str(e)}")
            raise
    
    def _extract_from_pdf(self, file_stream: BinaryIO) -> str:
        """
        Extract text from PDF
        
        Args:
            file_stream: File stream
            
        Returns:
            Extracted text
        """
        try:
            pdf_reader = PyPDF2.PdfReader(file_stream)
            text = ""
            
            # Extract text from each page
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n\n"
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {str(e)}")
            raise
    
    def _extract_from_docx(self, file_stream: BinaryIO) -> str:
        """
        Extract text from DOCX
        
        Args:
            file_stream: File stream
            
        Returns:
            Extracted text
        """
        try:
            doc = docx.Document(file_stream)
            text = ""
            
            # Extract text from paragraphs
            for para in doc.paragraphs:
                text += para.text + "\n"
                
            # Extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        text += cell.text + " "
                    text += "\n"
                text += "\n"
                
            return text.strip()
            
        except Exception as e:
            logger.error(f"Error extracting text from DOCX: {str(e)}")
            raise