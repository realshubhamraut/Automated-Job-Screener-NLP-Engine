"""
Azure Storage utilities for document management.
Handles uploading, downloading and managing documents in Azure Blob Storage.
"""

import os
import json
import uuid
from io import BytesIO
from datetime import datetime, timedelta
from typing import Dict, List, Optional, BinaryIO, Union, Tuple

import azure.core.exceptions
from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, generate_blob_sas, BlobSasPermissions
from azure.identity import DefaultAzureCredential

from src.utils.logger import get_logger
from src.config import (
    AZURE_STORAGE_ENABLED, AZURE_STORAGE_CONNECTION_STRING, 
    AZURE_STORAGE_ACCOUNT_NAME, AZURE_STORAGE_ACCOUNT_KEY,
    AZURE_STORAGE_USE_CONNECTION_STRING, AZURE_STORAGE_USE_SAS_TOKEN,
    AZURE_STORAGE_SAS_TOKEN, AZURE_CONTAINER_RAW_RESUMES,
    AZURE_CONTAINER_RAW_JOBS, AZURE_CONTAINER_PROCESSED_RESUMES,
    AZURE_CONTAINER_PROCESSED_JOBS
)

logger = get_logger(__name__)


class AzureStorageManager:
    """
    Handles Azure Storage operations for document management including:
    - Resume and job description storage
    - Raw and processed document management
    - Temporary access link generation
    """
    
    def __init__(self, connection_string: Optional[str] = None, 
                account_name: Optional[str] = None,
                use_connection_string: bool = True):
        """
        Initialize the Azure Storage Manager.
        
        Args:
            connection_string: Azure Storage connection string (optional, will use config if not provided)
            account_name: Azure Storage account name (optional, will use config if not provided)
            use_connection_string: Whether to use connection string or default credentials
        """
        self.connection_string = connection_string or AZURE_STORAGE_CONNECTION_STRING
        self.account_name = account_name or AZURE_STORAGE_ACCOUNT_NAME
        self.use_connection_string = use_connection_string if connection_string else AZURE_STORAGE_USE_CONNECTION_STRING
        
        self.containers = {
            "raw_resumes": AZURE_CONTAINER_RAW_RESUMES,
            "raw_jobs": AZURE_CONTAINER_RAW_JOBS,
            "processed_resumes": AZURE_CONTAINER_PROCESSED_RESUMES,
            "processed_jobs": AZURE_CONTAINER_PROCESSED_JOBS,
        }
        
        # Initialize Azure Blob Service client
        self._initialize_client()
        
        # Ensure containers exist
        self._ensure_containers_exist()
    
    def _initialize_client(self):
        """Initialize the Azure Blob Service client based on configuration."""
        try:
            if self.use_connection_string and self.connection_string:
                self.blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
                logger.info(f"Azure Storage client initialized using connection string")
            elif AZURE_STORAGE_USE_SAS_TOKEN and AZURE_STORAGE_SAS_TOKEN:
                sas_url = f"https://{self.account_name}.blob.core.windows.net/?{AZURE_STORAGE_SAS_TOKEN}"
                self.blob_service_client = BlobServiceClient(account_url=sas_url)
                logger.info(f"Azure Storage client initialized using SAS token")
            elif self.account_name:
                # Use DefaultAzureCredential for managed identities or service principals
                account_url = f"https://{self.account_name}.blob.core.windows.net"
                self.blob_service_client = BlobServiceClient(
                    account_url=account_url, 
                    credential=DefaultAzureCredential()
                )
                logger.info(f"Azure Storage client initialized using default credentials")
            else:
                raise ValueError("Either connection string, SAS token, or account name with managed identity must be provided")
        
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage client: {str(e)}")
            raise
    
    def _ensure_containers_exist(self):
        """Create containers if they don't exist."""
        try:
            for container_name in self.containers.values():
                container_client = self.blob_service_client.get_container_client(container_name)
                
                if not container_client.exists():
                    container_client.create_container()
                    logger.info(f"Created container {container_name}")
        
        except Exception as e:
            logger.error(f"Error ensuring containers exist: {str(e)}")
            raise
    
    def upload_resume(self, file_content: Union[bytes, BinaryIO], filename: str, 
                    content_type: str, is_processed: bool = False) -> str:
        """
        Upload a resume to Azure Blob Storage.
        
        Args:
            file_content: File content as bytes or file-like object
            filename: Original filename
            content_type: MIME type of the file
            is_processed: Whether this is a processed document
            
        Returns:
            Blob ID (UUID) used to identify the document in Azure
        """
        container_name = self.containers["processed_resumes"] if is_processed else self.containers["raw_resumes"]
        blob_id = str(uuid.uuid4())
        blob_name = f"{blob_id}/{os.path.basename(filename)}"
        
        try:
            # Create a blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Add metadata
            metadata = {
                "original_filename": filename,
                "content_type": content_type,
                "upload_date": datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            }
            
            # Upload the file
            if isinstance(file_content, bytes):
                blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)
            else:
                blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)
            
            logger.info(f"Uploaded resume {blob_name} to container {container_name}")
            return blob_id
            
        except Exception as e:
            logger.error(f"Error uploading resume {filename}: {str(e)}")
            raise
    
    def upload_job_description(self, file_content: Union[bytes, BinaryIO], filename: str, 
                             content_type: str, is_processed: bool = False) -> str:
        """
        Upload a job description to Azure Blob Storage.
        
        Args:
            file_content: File content as bytes or file-like object
            filename: Original filename
            content_type: MIME type of the file
            is_processed: Whether this is a processed document
            
        Returns:
            Blob ID (UUID) used to identify the document in Azure
        """
        container_name = self.containers["processed_jobs"] if is_processed else self.containers["raw_jobs"]
        blob_id = str(uuid.uuid4())
        blob_name = f"{blob_id}/{os.path.basename(filename)}"
        
        try:
            # Create a blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Add metadata
            metadata = {
                "original_filename": filename,
                "content_type": content_type,
                "upload_date": datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
            }
            
            # Upload the file
            if isinstance(file_content, bytes):
                blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)
            else:
                blob_client.upload_blob(file_content, overwrite=True, metadata=metadata)
            
            logger.info(f"Uploaded job description {blob_name} to container {container_name}")
            return blob_id
            
        except Exception as e:
            logger.error(f"Error uploading job description {filename}: {str(e)}")
            raise
    
    def download_document(self, doc_id: str, doc_type: str, is_processed: bool = False) -> Tuple[bytes, Dict]:
        """
        Download a document from Azure Blob Storage.
        
        Args:
            doc_id: Document ID (blob ID)
            doc_type: Type of document ('resume' or 'job')
            is_processed: Whether this is a processed document
            
        Returns:
            Tuple of (file content as bytes, metadata dictionary)
        """
        if doc_type.lower() == 'resume':
            container_name = self.containers["processed_resumes"] if is_processed else self.containers["raw_resumes"]
        elif doc_type.lower() == 'job':
            container_name = self.containers["processed_jobs"] if is_processed else self.containers["raw_jobs"]
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        try:
            # List blobs to find the one with the right prefix
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = list(container_client.list_blobs(name_starts_with=doc_id))
            
            if not blobs:
                raise ValueError(f"Document {doc_id} not found in container {container_name}")
            
            # Get the first matching blob
            blob_name = blobs[0].name
            blob_client = self.blob_service_client.get_blob_client(container=container_name, blob=blob_name)
            
            # Download content and metadata
            download_stream = blob_client.download_blob()
            content = download_stream.readall()
            metadata = blob_client.get_blob_properties().metadata
            
            return content, metadata
            
        except Exception as e:
            logger.error(f"Error downloading document {doc_id}: {str(e)}")
            raise
    
    def delete_document(self, doc_id: str, doc_type: str, delete_processed: bool = True) -> bool:
        """
        Delete a document from Azure Blob Storage.
        
        Args:
            doc_id: Document ID (blob ID)
            doc_type: Type of document ('resume' or 'job')
            delete_processed: Whether to also delete the processed version
            
        Returns:
            True if successful, False otherwise
        """
        success = True
        
        # Delete raw document
        if doc_type.lower() == 'resume':
            raw_container = self.containers["raw_resumes"]
            processed_container = self.containers["processed_resumes"]
        elif doc_type.lower() == 'job':
            raw_container = self.containers["raw_jobs"]
            processed_container = self.containers["processed_jobs"]
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        try:
            # Delete from raw container
            container_client = self.blob_service_client.get_container_client(raw_container)
            blobs = list(container_client.list_blobs(name_starts_with=doc_id))
            
            for blob in blobs:
                blob_client = self.blob_service_client.get_blob_client(container=raw_container, blob=blob.name)
                blob_client.delete_blob()
                logger.info(f"Deleted raw {doc_type} {blob.name} from container {raw_container}")
            
            # Delete processed document if requested
            if delete_processed:
                # Delete JSON from processed container
                processed_blob_name = f"{doc_id}.json"
                blob_client = self.blob_service_client.get_blob_client(
                    container=processed_container, 
                    blob=processed_blob_name
                )
                
                # Only delete if exists
                try:
                    blob_client.delete_blob()
                    logger.info(f"Deleted processed {doc_type} {doc_id} from container {processed_container}")
                except azure.core.exceptions.ResourceNotFoundError:
                    logger.warning(f"Processed {doc_type} {doc_id} not found in container {processed_container}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error deleting {doc_type} {doc_id}: {str(e)}")
            return False
    
    def store_processed_document(self, doc_id: str, doc_type: str, data: Dict) -> bool:
        """
        Store processed document data in Azure Storage.
        
        Args:
            doc_id: Document ID
            doc_type: Type of document ('resume' or 'job')
            data: Processed data to store (will be serialized to JSON)
            
        Returns:
            True if successful, False otherwise
        """
        if doc_type.lower() == 'resume':
            container_name = self.containers["processed_resumes"]
        elif doc_type.lower() == 'job':
            container_name = self.containers["processed_jobs"]
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        blob_name = f"{doc_id}.json"
        
        try:
            # Convert data to JSON
            json_data = json.dumps(data, ensure_ascii=False).encode('utf-8')
            
            # Upload JSON to blob storage
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            blob_client.upload_blob(json_data, overwrite=True, content_settings={
                "content_type": "application/json;charset=utf-8"
            })
            
            logger.info(f"Stored processed {doc_type} {doc_id} in Azure Storage")
            return True
            
        except Exception as e:
            logger.error(f"Error storing processed {doc_type} {doc_id}: {str(e)}")
            return False
    
    def get_processed_document(self, doc_id: str, doc_type: str) -> Optional[Dict]:
        """
        Get processed document data from Azure Storage.
        
        Args:
            doc_id: Document ID
            doc_type: Type of document ('resume' or 'job')
            
        Returns:
            Dictionary of processed data or None if not found
        """
        if doc_type.lower() == 'resume':
            container_name = self.containers["processed_resumes"]
        elif doc_type.lower() == 'job':
            container_name = self.containers["processed_jobs"]
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        blob_name = f"{doc_id}.json"
        
        try:
            # Get the blob client
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Download and parse JSON data
            download_stream = blob_client.download_blob()
            json_content = download_stream.readall().decode('utf-8')
            data = json.loads(json_content)
            
            return data
            
        except azure.core.exceptions.ResourceNotFoundError:
            logger.warning(f"Processed {doc_type} {doc_id} not found in Azure Storage")
            return None
        except Exception as e:
            logger.error(f"Error retrieving processed {doc_type} {doc_id}: {str(e)}")
            return None
    
    def list_documents(self, doc_type: str, processed_only: bool = False) -> List[Dict]:
        """
        List documents in Azure Storage.
        
        Args:
            doc_type: Type of document ('resume' or 'job')
            processed_only: Whether to only list processed documents
            
        Returns:
            List of dictionaries with document information
        """
        if doc_type.lower() == 'resume':
            container_name = self.containers["processed_resumes"] if processed_only else self.containers["raw_resumes"]
        elif doc_type.lower() == 'job':
            container_name = self.containers["processed_jobs"] if processed_only else self.containers["raw_jobs"]
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        try:
            # Get container client
            container_client = self.blob_service_client.get_container_client(container_name)
            
            # List all blobs in the container
            blobs = list(container_client.list_blobs())
            
            results = []
            for blob in blobs:
                # For processed documents, only include JSON files
                if processed_only and not blob.name.endswith('.json'):
                    continue
                
                # Get blob client to access metadata
                blob_client = self.blob_service_client.get_blob_client(
                    container=container_name,
                    blob=blob.name
                )
                
                # Get blob properties with metadata
                properties = blob_client.get_blob_properties()
                
                # Extract document ID from the blob name
                doc_id = blob.name.split('/')[0] if '/' in blob.name else blob.name.split('.')[0]
                
                # Add to results
                results.append({
                    'id': doc_id,
                    'filename': properties.metadata.get('original_filename', blob.name),
                    'upload_time': properties.metadata.get('upload_date', ''),
                    'size': blob.size,
                    'content_type': properties.metadata.get('content_type', ''),
                    'last_modified': blob.last_modified
                })
            
            return results
            
        except Exception as e:
            logger.error(f"Error listing {doc_type} documents: {str(e)}")
            return []
    
    def generate_sas_url(self, doc_id: str, doc_type: str, is_processed: bool = False,
                       expiry_minutes: int = 60) -> Optional[str]:
        """
        Generate a SAS URL for temporary access to a blob.
        
        Args:
            doc_id: Document ID
            doc_type: Type of document ('resume' or 'job')
            is_processed: Whether this is a processed document
            expiry_minutes: Minutes until the SAS URL expires
            
        Returns:
            SAS URL for the blob or None if failed
        """
        if not AZURE_STORAGE_ACCOUNT_KEY:
            logger.error("Cannot generate SAS URL without account key")
            return None
            
        if doc_type.lower() == 'resume':
            container_name = self.containers["processed_resumes"] if is_processed else self.containers["raw_resumes"]
        elif doc_type.lower() == 'job':
            container_name = self.containers["processed_jobs"] if is_processed else self.containers["raw_jobs"]
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        try:
            # Find the blob
            container_client = self.blob_service_client.get_container_client(container_name)
            blobs = list(container_client.list_blobs(name_starts_with=doc_id))
            
            if not blobs:
                logger.warning(f"Document {doc_id} not found in container {container_name}")
                return None
                
            # Get the blob name
            blob_name = blobs[0].name
            
            # Calculate expiry time
            expiry_time = datetime.utcnow() + timedelta(minutes=expiry_minutes)
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.account_name,
                account_key=AZURE_STORAGE_ACCOUNT_KEY,
                container_name=container_name,
                blob_name=blob_name,
                permission=BlobSasPermissions(read=True),
                expiry=expiry_time
            )
            
            # Construct the full URL
            sas_url = f"https://{self.account_name}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
            logger.info(f"Generated SAS URL for {doc_type} {doc_id} valid for {expiry_minutes} minutes")
            
            return sas_url
            
        except Exception as e:
            logger.error(f"Error generating SAS URL for {doc_type} {doc_id}: {str(e)}")
            return None
            
    def clear_all_documents(self, doc_type: str = None) -> bool:
        """
        Clear all documents from Azure Storage.
        
        Args:
            doc_type: Optional document type to clear ('resume' or 'job')
                     If None, clears all documents
        
        Returns:
            True if successful, False otherwise
        """
        containers_to_clear = []
        
        if doc_type is None:
            # Clear all containers
            containers_to_clear = list(self.containers.values())
        elif doc_type.lower() == 'resume':
            containers_to_clear = [self.containers["raw_resumes"], self.containers["processed_resumes"]]
        elif doc_type.lower() == 'job':
            containers_to_clear = [self.containers["raw_jobs"], self.containers["processed_jobs"]]
        else:
            raise ValueError(f"Invalid document type: {doc_type}")
        
        try:
            for container_name in containers_to_clear:
                container_client = self.blob_service_client.get_container_client(container_name)
                
                # List all blobs in the container
                blobs = list(container_client.list_blobs())
                
                # Delete each blob
                for blob in blobs:
                    container_client.delete_blob(blob.name)
                    
            logger.info(f"Cleared all documents from Azure Storage")
            return True
            
        except Exception as e:
            logger.error(f"Error clearing documents from Azure Storage: {str(e)}")
            return False