import os
import json
import numpy as np
import faiss
import pickle
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import uuid

from src.vector_store.base import VectorStore
from src.document_processor.embedder import TextEmbedder
from src.config import VECTOR_STORE_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

class FAISSStore(VectorStore):
    """FAISS vector store implementation"""
    
    def __init__(self, dimension: int = 384, store_name: str = "default", embedding_model: Optional[str] = None):
        """
        Initialize the FAISS vector store
        
        Args:
            dimension (int): Dimension of the embeddings
            store_name (str): Name of the vector store
            embedding_model (str, optional): Name of the embedding model
        """
        self.dimension = dimension
        self.store_name = store_name
        self.index = faiss.IndexFlatIP(dimension)  # Inner product index (cosine similarity)
        self.documents = {}  # Document storage: {doc_id: document}
        self.doc_ids = []  # List of doc_ids in order matching the index
        self.index_path = os.path.join(VECTOR_STORE_DIR, f"{store_name}_faiss.index")
        self.docs_path = os.path.join(VECTOR_STORE_DIR, f"{store_name}_docs.pkl")
        
        # Initialize embedder if model name provided
        self.embedder = TextEmbedder(model_name=embedding_model) if embedding_model else None
        
    def add_document(self, doc_id: str, document: Dict[str, Any], embedding: np.ndarray) -> bool:
        """
        Add a document and its embedding to the vector store
        
        Args:
            doc_id (str): Unique identifier for the document
            document (dict): Document content and metadata
            embedding (np.ndarray): Vector representation of the document
            
        Returns:
            bool: Success status
        """
        try:
            # Ensure the embedding is the right shape and type
            embedding = np.array(embedding).astype('float32').reshape(1, -1)
            
            # Add to FAISS index
            self.index.add(embedding)
            
            # Store document
            self.documents[doc_id] = document
            self.doc_ids.append(doc_id)
            
            logger.info(f"Added document {doc_id} to FAISS store")
            return True
        except Exception as e:
            logger.error(f"Error adding document to FAISS store: {str(e)}")
            return False
    
    def add_documents(self, documents: List[Tuple[str, Dict[str, Any]]], embeddings: List[np.ndarray]) -> bool:
        """
        Add multiple documents and their embeddings to the vector store
        
        Args:
            documents (list): List of (doc_id, document) tuples
            embeddings (list): List of vector representations
            
        Returns:
            bool: Success status
        """
        try:
            # Convert embeddings to numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            # Add to FAISS index
            self.index.add(embeddings_array)
            
            # Store documents
            for (doc_id, document), _ in zip(documents, embeddings):
                self.documents[doc_id] = document
                self.doc_ids.append(doc_id)
            
            logger.info(f"Added {len(documents)} documents to FAISS store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to FAISS store: {str(e)}")
            return False
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a document by ID
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            dict: Document content and metadata
        """
        return self.documents.get(doc_id)
    
    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for similar documents
        
        Args:
            query_embedding (np.ndarray): Vector representation of the query
            top_k (int): Number of results to return
            
        Returns:
            list: Top k matching documents with similarity scores as
                [(doc_id, document, score), ...]
        """
        try:
            # Ensure the query embedding is the right shape and type
            query_embedding = np.array(query_embedding).astype('float32').reshape(1, -1)
            
            # Search for similar documents
            k = min(top_k, len(self.doc_ids))
            if k == 0:
                return []
                
            scores, indices = self.index.search(query_embedding, k)
            
            # Format results
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.doc_ids) and idx >= 0:
                    doc_id = self.doc_ids[idx]
                    document = self.documents.get(doc_id)
                    if document:
                        results.append((doc_id, document, float(score)))
                        
            return results
        except Exception as e:
            logger.error(f"Error searching in FAISS store: {str(e)}")
            return []
    
    def search_by_text(self, query_text: str, top_k: int = 5) -> List[Tuple[str, Dict[str, Any], float]]:
        """
        Search for documents by text query
        
        Args:
            query_text (str): Text query
            top_k (int): Number of results to return
            
        Returns:
            list: Top k matching documents with similarity scores as
                [(doc_id, document, score), ...]
        """
        if not self.embedder:
            raise ValueError("No embedder provided. Initialize with embedding_model or set embedder.")
            
        # Generate embedding for query
        query_embedding = self.embedder.embed_text(query_text)
        
        # Use the embedding to search
        return self.search(query_embedding, top_k)
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store
        
        Note: FAISS doesn't support direct deletion, so we need to rebuild the index
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            bool: Success status
        """
        try:
            # Check if document exists
            if doc_id not in self.documents:
                logger.warning(f"Document {doc_id} not found in store")
                return False
            
            # Remove document
            del self.documents[doc_id]
            
            # Find the index of the document ID
            try:
                idx = self.doc_ids.index(doc_id)
                self.doc_ids.pop(idx)
            except ValueError:
                logger.warning(f"Document ID {doc_id} not found in doc_ids list")
            
            # Rebuild index (FAISS doesn't support direct deletion)
            # This is inefficient but necessary
            self.rebuild_index()
            
            logger.info(f"Deleted document {doc_id} from FAISS store")
            return True
        except Exception as e:
            logger.error(f"Error deleting document from FAISS store: {str(e)}")
            return False
    
    def rebuild_index(self) -> bool:
        """
        Rebuild the FAISS index from stored documents
        
        Returns:
            bool: Success status
        """
        try:
            # Create a new index
            self.index = faiss.IndexFlatIP(self.dimension)
            
            # If there are no documents, we're done
            if not self.documents:
                logger.info("No documents to rebuild index with")
                return True
            
            if not self.embedder:
                raise ValueError("No embedder provided. Cannot rebuild index without embeddings.")
            
            # Re-add all documents
            new_doc_ids = []
            embeddings = []
            
            for doc_id in self.doc_ids:
                document = self.documents.get(doc_id)
                if document:
                    # Re-embed the document
                    if 'original_text' in document:
                        embedding = self.embedder.embed_text(document['original_text'])
                        embeddings.append(embedding)
                        new_doc_ids.append(doc_id)
            
            # Convert embeddings to numpy array
            if embeddings:
                embeddings_array = np.array(embeddings).astype('float32')
                
                # Add to FAISS index
                self.index.add(embeddings_array)
                
                # Update doc_ids
                self.doc_ids = new_doc_ids
                
            logger.info(f"Rebuilt FAISS index with {len(self.doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error rebuilding FAISS index: {str(e)}")
            return False
    
    def clear(self) -> bool:
        """
        Clear all documents from the vector store
        
        Returns:
            bool: Success status
        """
        try:
            self.index = faiss.IndexFlatIP(self.dimension)
            self.documents = {}
            self.doc_ids = []
            logger.info("Cleared FAISS store")
            return True
        except Exception as e:
            logger.error(f"Error clearing FAISS store: {str(e)}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store
        
        Returns:
            dict: Statistics like document count, dimensions, etc.
        """
        return {
            'store_name': self.store_name,
            'dimensions': self.dimension,
            'document_count': len(self.doc_ids),
            'unique_document_count': len(self.documents),
            'index_type': 'FAISS IndexFlatIP'
        }
    
    def save(self, path: Optional[str] = None) -> bool:
        """
        Save the vector store to disk
        
        Args:
            path (str, optional): Directory path to save the vector store
            
        Returns:
            bool: Success status
        """
        try:
            # Create directory if it doesn't exist
            save_dir = path or VECTOR_STORE_DIR
            os.makedirs(save_dir, exist_ok=True)
            
            index_path = os.path.join(save_dir, f"{self.store_name}_faiss.index")
            docs_path = os.path.join(save_dir, f"{self.store_name}_docs.pkl")
            metadata_path = os.path.join(save_dir, f"{self.store_name}_metadata.json")
            
            # Save FAISS index
            faiss.write_index(self.index, index_path)
            
            # Save documents and doc_ids
            with open(docs_path, 'wb') as f:
                pickle.dump({'documents': self.documents, 'doc_ids': self.doc_ids}, f)
            
            # Save metadata
            metadata = {
                'dimension': self.dimension,
                'store_name': self.store_name,
                'document_count': len(self.doc_ids),
                'created_at': datetime.now().isoformat(),
                'index_type': 'FAISS IndexFlatIP'
            }
            
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f)
            
            logger.info(f"Saved FAISS store to {save_dir}")
            return True
        except Exception as e:
            logger.error(f"Error saving FAISS store: {str(e)}")
            return False
    
    def load(self, path: Optional[str] = None) -> bool:
        """
        Load the vector store from disk
        
        Args:
            path (str, optional): Directory path to load the vector store from
            
        Returns:
            bool: Success status
        """
        try:
            load_dir = path or VECTOR_STORE_DIR
            index_path = os.path.join(load_dir, f"{self.store_name}_faiss.index")
            docs_path = os.path.join(load_dir, f"{self.store_name}_docs.pkl")
            
            # Check if files exist
            if not os.path.exists(index_path) or not os.path.exists(docs_path):
                logger.warning(f"FAISS store files not found at {load_dir}")
                return False
            
            # Load FAISS index
            self.index = faiss.read_index(index_path)
            
            # Load documents and doc_ids
            with open(docs_path, 'rb') as f:
                data = pickle.load(f)
                self.documents = data['documents']
                self.doc_ids = data['doc_ids']
            
            logger.info(f"Loaded FAISS store from {load_dir} with {len(self.doc_ids)} documents")
            return True
        except Exception as e:
            logger.error(f"Error loading FAISS store: {str(e)}")
            return False