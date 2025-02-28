import torch
import numpy as np
from typing import List, Dict, Any, Union, Optional
from transformers import AutoTokenizer, AutoModel
import os

from src.config import EMBEDDING_MODEL
from src.utils.logger import get_logger

logger = get_logger(__name__)

class TextEmbedder:
    """Generate embeddings for text using PyTorch models directly."""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL, device: Optional[str] = None):
        """
        Initialize the embedder with a model
        
        Args:
            model_name (str): Name of the transformer model to use
            device (str, optional): Device to use for computation ('cpu', 'cuda')
        """
        self.model_name = model_name
        
        # Set device for PyTorch
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"Initializing embedder with model {model_name} on {self.device}")
        
        try:
            # Load tokenizer and model directly from Hugging Face
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)
            logger.info(f"Model loaded successfully: {model_name}")
            
            # Add the dimension attribute - this was missing in the original implementation
            self._dimension = self.model.config.hidden_size
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    @property
    def dimension(self):
        """Get the dimension of embeddings produced by this model"""
        return self._dimension
    
    def _mean_pooling(self, model_output, attention_mask):
        """
        Perform mean pooling on token embeddings
        
        Args:
            model_output: Output from the model
            attention_mask: Attention mask from tokenizer
            
        Returns:
            Pooled embeddings
        """
        # First element of model_output contains all token embeddings
        token_embeddings = model_output[0]
        
        # Expand attention mask from [batch_size, seq_length] to [batch_size, seq_length, hidden_size]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        
        # Sum token embeddings and divide by the expanded attention mask
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        
        # Return mean pooled embeddings
        return sum_embeddings / sum_mask
    
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for a single text
        
        Args:
            text (str): Text to embed
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not text or not isinstance(text, str):
            logger.warning("Invalid text for embedding")
            # Return zero vector of appropriate dimension
            return np.zeros(self.model.config.hidden_size)
        
        try:
            # Tokenize and prepare input
            encoded_input = self.tokenizer(
                text, 
                padding=True, 
                truncation=True, 
                max_length=512, 
                return_tensors='pt'
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)
            
            # Perform pooling and get embeddings
            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
            
            # Normalize embeddings
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            # Convert to numpy and return the first (only) embedding
            return embeddings[0].cpu().numpy()
        
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return np.zeros(self.model.config.hidden_size)
    
    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for a list of texts
        
        Args:
            texts (List[str]): List of texts to embed
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        if not texts:
            logger.warning("Empty list of texts for embedding")
            return np.array([])
        
        # Filter out invalid texts
        valid_texts = [t for t in texts if isinstance(t, str) and t.strip()]
        
        if not valid_texts:
            logger.warning("No valid texts for embedding")
            return np.zeros((0, self.model.config.hidden_size))
        
        try:
            # Process in smaller batches to avoid OOM issues
            batch_size = 32
            all_embeddings = []
            
            for i in range(0, len(valid_texts), batch_size):
                batch_texts = valid_texts[i:i+batch_size]
                
                # Tokenize and prepare input
                encoded_input = self.tokenizer(
                    batch_texts, 
                    padding=True, 
                    truncation=True, 
                    max_length=512, 
                    return_tensors='pt'
                ).to(self.device)
                
                # Generate embeddings
                with torch.no_grad():
                    model_output = self.model(**encoded_input)
                
                # Perform pooling and get embeddings
                embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])
                
                # Normalize embeddings
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Add to list
                all_embeddings.append(embeddings.cpu().numpy())
            
            # Combine batch results
            return np.vstack(all_embeddings)
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            return np.zeros((len(valid_texts), self.model.config.hidden_size))
    
    def embed_document(self, document: Dict[str, Any], text_field: str = 'processed.clean_text') -> np.ndarray:
        """
        Generate embedding for a document
        
        Args:
            document (Dict[str, Any]): Document to embed
            text_field (str): Field containing the text to embed (supports dot notation)
            
        Returns:
            np.ndarray: Embedding vector
        """
        # Extract text from document using the field path
        text = document
        for field in text_field.split('.'):
            if isinstance(text, dict) and field in text:
                text = text[field]
            else:
                logger.warning(f"Field '{field}' not found in document")
                return np.zeros(self.model.config.hidden_size)
        
        return self.embed_text(text)
    
    def embed_documents(self, documents: List[Dict[str, Any]], text_field: str = 'processed.clean_text') -> np.ndarray:
        """
        Generate embeddings for a list of documents
        
        Args:
            documents (List[Dict[str, Any]]): List of documents to embed
            text_field (str): Field containing the text to embed (supports dot notation)
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        texts = []
        for doc in documents:
            # Extract text from document using the field path
            text = doc
            for field in text_field.split('.'):
                if isinstance(text, dict) and field in text:
                    text = text[field]
                else:
                    text = ""
                    break
            texts.append(text)
        
        return self.embed_texts(texts)
    
    def embed_chunks(self, chunks: List[str]) -> np.ndarray:
        """
        Generate embeddings for text chunks
        
        Args:
            chunks (List[str]): Text chunks to embed
            
        Returns:
            np.ndarray: Array of embedding vectors
        """
        return self.embed_texts(chunks)