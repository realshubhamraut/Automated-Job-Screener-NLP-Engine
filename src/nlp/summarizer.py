import os
import torch
from typing import Dict, Any, List, Optional
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import SUMMARIZATION_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TextSummarizer:
    """Generate summaries of text documents using transformer models"""
    
    def __init__(self, model_name: str = SUMMARIZATION_MODEL, device: Optional[str] = None):
        """
        Initialize the summarizer with a transformer model
        
        Args:
            model_name (str): Name of the transformer model to use
            device (str, optional): Device to run model on ('cpu', 'cuda', etc.)
        """
        self.model_name = model_name
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        logger.info(f"Initializing summarizer with model {model_name} on {self.device}")
        
        try:
            # Initialize the summarization pipeline
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            self.summarizer = pipeline("summarization", model=self.model, tokenizer=self.tokenizer, device=self.device)
            logger.info("Summarizer initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing summarizer: {str(e)}")
            raise
    
    def _split_text_for_long_input(self, text: str, max_length: int = 1024) -> List[str]:
        """
        Split text into chunks for long inputs
        
        Args:
            text (str): Text to split
            max_length (int): Maximum chunk length in tokens
            
        Returns:
            List[str]: List of text chunks
        """
        # Tokenize the text
        tokens = self.tokenizer.tokenize(text)
        
        # Split into chunks
        chunks = []
        current_chunk = []
        current_chunk_length = 0
        
        for token in tokens:
            if current_chunk_length + 1 <= max_length:
                current_chunk.append(token)
                current_chunk_length += 1
            else:
                # Chunk is full, detokenize and store
                chunk_text = self.tokenizer.convert_tokens_to_string(current_chunk)
                chunks.append(chunk_text)
                
                # Start new chunk with current token
                current_chunk = [token]
                current_chunk_length = 1
        
        # Add final chunk if exists
        if current_chunk:
            chunk_text = self.tokenizer.convert_tokens_to_string(current_chunk)
            chunks.append(chunk_text)
        
        return chunks
    
    def summarize_text(self, text: str, max_length: int = 150, min_length: int = 50, 
                      do_sample: bool = False) -> str:
        """
        Generate a summary of the input text
        
        Args:
            text (str): Text to summarize
            max_length (int): Maximum length of the summary
            min_length (int): Minimum length of the summary
            do_sample (bool): Whether to use sampling for generation
            
        Returns:
            str: Generated summary
        """
        try:
            if not text or not isinstance(text, str) or len(text.strip()) < 100:
                logger.warning("Text too short or invalid for summarization")
                return text
            
            # Check if text is too long for the model
            input_ids = self.tokenizer.encode(text, return_tensors="pt")
            max_input_length = self.model.config.max_position_embeddings
            
            if input_ids.shape[1] > max_input_length:
                logger.info(f"Text too long ({input_ids.shape[1]} tokens), splitting into chunks")
                
                # Split text into chunks
                text_chunks = self._split_text_for_long_input(text, max_input_length - 100)
                
                # Summarize each chunk
                chunk_summaries = []
                for chunk in text_chunks:
                    try:
                        chunk_summary = self.summarizer(chunk, max_length=max_length // len(text_chunks), 
                                                      min_length=min_length // len(text_chunks), 
                                                      do_sample=do_sample)[0]['summary_text']
                        chunk_summaries.append(chunk_summary)
                    except Exception as e:
                        logger.error(f"Error summarizing text chunk: {str(e)}")
                
                # Combine chunk summaries
                combined_summary = " ".join(chunk_summaries)
                
                # If combined summary is still too long, summarize again
                if len(combined_summary) > max_length * 2:
                    logger.info("Combined chunk summaries still too long, summarizing again")
                    try:
                        final_summary = self.summarizer(combined_summary, max_length=max_length, 
                                                      min_length=min_length, 
                                                      do_sample=do_sample)[0]['summary_text']
                        return final_summary
                    except Exception as e:
                        logger.error(f"Error summarizing combined chunks: {str(e)}")
                        return combined_summary
                else:
                    return combined_summary
            else:
                # Text is within model limits
                summary = self.summarizer(text, max_length=max_length, min_length=min_length, 
                                        do_sample=do_sample)[0]['summary_text']
                return summary
        except Exception as e:
            logger.error(f"Error summarizing text: {str(e)}")
            # Return a truncated version of the original text as fallback
            return text[:max_length * 2] + "..."
    
    def summarize_document(self, document: Dict[str, Any], text_field: str = 'original_text', 
                          max_length: int = 150) -> str:
        """
        Generate a summary of a document
        
        Args:
            document (Dict[str, Any]): Document to summarize
            text_field (str): Field containing the text to summarize
            max_length (int): Maximum length of the summary
            
        Returns:
            str: Generated summary
        """
        if text_field not in document:
            logger.warning(f"Text field '{text_field}' not found in document")
            return ""
        
        return self.summarize_text(document[text_field], max_length=max_length)
    
    def add_summary_to_document(self, document: Dict[str, Any], text_field: str = 'original_text', 
                              max_length: int = 150) -> Dict[str, Any]:
        """
        Add a summary to a document
        
        Args:
            document (Dict[str, Any]): Document to summarize
            text_field (str): Field containing the text to summarize
            max_length (int): Maximum length of the summary
            
        Returns:
            Dict[str, Any]: Document with added summary
        """
        document['summary'] = self.summarize_document(document, text_field, max_length)
        return document
    
    # Shortcut method name for convenience
    def summarize(self, text_or_doc, **kwargs):
        """
        Summarize text or document
        
        Args:
            text_or_doc: Text string or document dict to summarize
            **kwargs: Additional arguments to pass to summarize_text or summarize_document
            
        Returns:
            str: Generated summary
        """
        if isinstance(text_or_doc, dict):
            return self.summarize_document(text_or_doc, **kwargs)
        else:
            return self.summarize_text(text_or_doc, **kwargs)