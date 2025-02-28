from abc import ABC, abstractmethod

class VectorStore(ABC):
    """Abstract base class for vector storage implementations"""
    
    @abstractmethod
    def add_document(self, doc_id, document, embedding):
        """
        Add a document and its embedding to the vector store
        
        Args:
            doc_id (str): Unique identifier for the document
            document (dict): Document content and metadata
            embedding (numpy.ndarray): Vector representation of the document
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def add_documents(self, documents, embeddings):
        """
        Add multiple documents and their embeddings to the vector store
        
        Args:
            documents (list): List of (doc_id, document) tuples
            embeddings (list): List of vector representations
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def get_document(self, doc_id):
        """
        Retrieve a document by ID
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            dict: Document content and metadata
        """
        pass
    
    @abstractmethod
    def search(self, query_embedding, top_k=5):
        """
        Search for similar documents
        
        Args:
            query_embedding (numpy.ndarray): Vector representation of the query
            top_k (int): Number of results to return
            
        Returns:
            list: Top k matching documents with similarity scores as
                [(doc_id, document, score), ...]
        """
        pass
    
    @abstractmethod
    def search_by_text(self, query_text, top_k=5):
        """
        Search for documents by text query
        
        Args:
            query_text (str): Text query
            top_k (int): Number of results to return
            
        Returns:
            list: Top k matching documents with similarity scores as
                [(doc_id, document, score), ...]
        """
        pass
    
    @abstractmethod
    def delete(self, doc_id):
        """
        Delete a document from the vector store
        
        Args:
            doc_id (str): Document ID
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def clear(self):
        """
        Clear all documents from the vector store
        
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def get_stats(self):
        """
        Get statistics about the vector store
        
        Returns:
            dict: Statistics like document count, dimensions, etc.
        """
        pass
    
    @abstractmethod
    def save(self, path=None):
        """
        Save the vector store to disk
        
        Args:
            path (str, optional): Path to save the vector store. If None, use default path.
            
        Returns:
            bool: Success status
        """
        pass
    
    @abstractmethod
    def load(self, path=None):
        """
        Load the vector store from disk
        
        Args:
            path (str, optional): Path to load the vector store from. If None, use default path.
            
        Returns:
            bool: Success status
        """
        pass