
from typing import List, Dict, Any, Tuple
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)

class HybridSearchEngine:
    """
    Combines semantic search (embeddings) and keyword search (TF-IDF)
    for more effective document matching
    """
    
    def __init__(self, hybrid_weight: float = 0.7, max_keywords: int = 20):
        """
        Initialize the hybrid search engine
        
        Args:
            hybrid_weight: Weight for semantic search (1.0 = only semantic, 0.0 = only keyword)
            max_keywords: Maximum number of keywords to extract from each document
        """
        self.hybrid_weight = hybrid_weight
        self.max_keywords = max_keywords
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1
        )
    
    def extract_keywords(self, text: str, top_n: int = 10) -> List[str]:
        """
        Extract keywords from text using TF-IDF
        
        Args:
            text: Text to extract keywords from
            top_n: Number of top keywords to return
            
        Returns:
            List of keywords
        """
        try:
            if not text or not isinstance(text, str):
                return []
                
            # Make sure the vectorizer is fit to the data before extraction
            self.vectorizer.fit([text])
            
            # Transform the text
            tfidf_matrix = self.vectorizer.transform([text])
            feature_names = self.vectorizer.get_feature_names_out()
            
            # Get the top n features by TF-IDF score
            dense = tfidf_matrix.todense()
            doc_dense = dense[0].tolist()[0]
            word_tfidf = [(word, doc_dense[idx]) for idx, word in enumerate(feature_names)]
            
            # Sort by score and return top words
            top_keywords = sorted(word_tfidf, key=lambda x: x[1], reverse=True)[:top_n]
            return [word for word, score in top_keywords]
            
        except Exception as e:
            logger.error(f"Error extracting keywords: {str(e)}")
            return []
    
    def calculate_keyword_similarity(self, source_keywords: List[str], target_keywords: List[str]) -> float:
        """
        Calculate keyword-based similarity between two sets of keywords
        
        Args:
            source_keywords: Keywords from source document
            target_keywords: Keywords from target document
            
        Returns:
            Similarity score (0-1)
        """
        if not source_keywords or not target_keywords:
            return 0.0
        
        # Convert to sets for intersection
        source_set = set(source_keywords)
        target_set = set(target_keywords)
        
        # Calculate Jaccard similarity
        intersection = len(source_set.intersection(target_set))
        union = len(source_set.union(target_set))
        
        if union == 0:
            return 0.0
            
        return intersection / union
    
    def hybrid_match(self, 
                    source_embedding: np.ndarray, 
                    target_embedding: np.ndarray,
                    source_text: str,
                    target_text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate hybrid similarity score combining embedding and keyword matching
        
        Args:
            source_embedding: Embedding of source document
            target_embedding: Embedding of target document
            source_text: Source document text
            target_text: Target document text
            
        Returns:
            Tuple of (hybrid_score, details_dict)
        """
        # Calculate semantic similarity using embeddings
        semantic_sim = self._semantic_similarity(source_embedding, target_embedding)
        
        # Extract keywords
        source_keywords = self.extract_keywords(source_text, self.max_keywords)
        target_keywords = self.extract_keywords(target_text, self.max_keywords)
        
        # Calculate keyword similarity
        keyword_sim = self.calculate_keyword_similarity(source_keywords, target_keywords)
        
        # Calculate weighted score
        hybrid_score = (self.hybrid_weight * semantic_sim) + ((1 - self.hybrid_weight) * keyword_sim)
        
        # Return score and details
        details = {
            'semantic_similarity': float(semantic_sim),
            'keyword_similarity': float(keyword_sim),
            'hybrid_weight': float(self.hybrid_weight),
            'hybrid_score': float(hybrid_score),
            'source_keywords': source_keywords,
            'target_keywords': target_keywords,
            'common_keywords': list(set(source_keywords).intersection(set(target_keywords)))
        }
        
        return hybrid_score, details
    
    def _semantic_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: First vector
            vec2: Second vector
            
        Returns:
            Cosine similarity (0-1)
        """
        if vec1 is None or vec2 is None:
            return 0.0
            
        if len(vec1.shape) == 1:
            vec1 = vec1.reshape(1, -1)
        if len(vec2.shape) == 1:
            vec2 = vec2.reshape(1, -1)
            
        try:
            return float(cosine_similarity(vec1, vec2)[0][0])
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            return 0.0