import os
import uuid
from django.db import models
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.utils.text import slugify
from django.utils import timezone


class DocumentStorage(FileSystemStorage):
    """Custom storage for document files to organize them in subfolders."""
    
    def get_available_name(self, name, max_length=None):
        """
        Generate a unique filename by adding a UUID to prevent overwrites.
        """
        dir_name, file_name = os.path.split(name)
        file_root, file_ext = os.path.splitext(file_name)
        
        # Clean up the filename
        file_root = slugify(file_root)
        
        # Add a UUID to make it unique
        unique_file = f"{file_root}_{uuid.uuid4().hex[:8]}{file_ext}"
        
        # Return the full path
        return os.path.join(dir_name, unique_file)


class Document(models.Model):
    """
    Base model for uploaded documents (resumes and job descriptions)
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Document types
    DOCUMENT_TYPES = [
        ('resume', 'Resume'),
        ('job_description', 'Job Description'),
    ]
    document_type = models.CharField(max_length=20, choices=DOCUMENT_TYPES)
    
    # File information
    original_filename = models.CharField(max_length=255)
    file = models.FileField(upload_to='uploads/documents/')
    file_size = models.IntegerField(help_text="File size in bytes")
    file_type = models.CharField(max_length=50, help_text="MIME type of the file")
    
    # Document content
    original_text = models.TextField(blank=True)
    clean_text = models.TextField(blank=True, help_text="Processed and cleaned text")
    
    # Metadata
    upload_date = models.DateTimeField(default=timezone.now)
    last_processed = models.DateTimeField(null=True, blank=True)
    is_processed = models.BooleanField(default=False)
    
    # Processing status
    PROCESSING_STATUS = [
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    processing_status = models.CharField(max_length=20, choices=PROCESSING_STATUS, default='pending')
    processing_error = models.TextField(blank=True)
    
    # User who uploaded the document
    uploaded_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, 
        on_delete=models.SET_NULL, 
        null=True,
        related_name='uploaded_documents'
    )
    
    class Meta:
        abstract = True
    
    def __str__(self):
        return self.original_filename


class Resume(Document):
    """
    Model for storing resume documents and their extracted information
    """
    # Resume-specific fields
    candidate_name = models.CharField(max_length=255, blank=True)
    candidate_email = models.EmailField(blank=True)
    candidate_phone = models.CharField(max_length=50, blank=True)
    candidate_location = models.CharField(max_length=255, blank=True)
    
    # Experience and education
    years_of_experience = models.FloatField(null=True, blank=True)
    current_job_title = models.CharField(max_length=255, blank=True)
    education_level = models.CharField(max_length=255, blank=True)
    
    # Extracted entities and sections
    extracted_entities = models.JSONField(default=dict, blank=True)
    extracted_sections = models.JSONField(default=dict, blank=True)
    extracted_skills = models.JSONField(default=list, blank=True)
    
    # Vector embedding for semantic search
    embedding_vector = models.BinaryField(blank=True, null=True)
    embedding_model = models.CharField(max_length=100, blank=True)
    
    # Document summary
    summary = models.TextField(blank=True)
    
    class Meta:
        verbose_name = "Resume"
        verbose_name_plural = "Resumes"
    
    def __str__(self):
        if self.candidate_name:
            return f"Resume: {self.candidate_name} - {self.original_filename}"
        return f"Resume: {self.original_filename}"


class JobDescription(Document):
    """
    Model for storing job description documents and their extracted information
    """
    # Job-specific fields
    job_title = models.CharField(max_length=255, blank=True)
    company_name = models.CharField(max_length=255, blank=True)
    job_location = models.CharField(max_length=255, blank=True)
    employment_type = models.CharField(max_length=100, blank=True)
    
    # Classification fields
    department = models.CharField(max_length=100, blank=True)
    seniority_level = models.CharField(max_length=50, blank=True)
    
    # Extracted entities and sections
    extracted_entities = models.JSONField(default=dict, blank=True)
    extracted_sections = models.JSONField(default=dict, blank=True)
    extracted_skills = models.JSONField(default=list, blank=True)
    required_skills = models.JSONField(default=list, blank=True)
    preferred_skills = models.JSONField(default=list, blank=True)
    
    # Vector embedding for semantic search
    embedding_vector = models.BinaryField(blank=True, null=True)
    embedding_model = models.CharField(max_length=100, blank=True)
    
    # Document summary
    summary = models.TextField(blank=True)
    
    class Meta:
        verbose_name = "Job Description"
        verbose_name_plural = "Job Descriptions"
    
    def __str__(self):
        if self.job_title:
            return f"Job: {self.job_title} - {self.original_filename}"
        return f"Job: {self.original_filename}"


class ProcessingTask(models.Model):
    """
    Model to track document processing tasks
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Task type
    TASK_TYPES = [
        ('resume_processing', 'Resume Processing'),
        ('job_processing', 'Job Description Processing'),
        ('embedding_generation', 'Embedding Generation'),
        ('entity_extraction', 'Entity Extraction'),
        ('skill_extraction', 'Skill Extraction'),
        ('section_extraction', 'Section Extraction'),
    ]
    task_type = models.CharField(max_length=30, choices=TASK_TYPES)
    
    # Connected to either Resume or Job Description
    resume = models.ForeignKey(Resume, on_delete=models.CASCADE, null=True, blank=True)
    job_description = models.ForeignKey(JobDescription, on_delete=models.CASCADE, null=True, blank=True)
    
    # Task status
    TASK_STATUS = [
        ('pending', 'Pending'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    ]
    status = models.CharField(max_length=20, choices=TASK_STATUS, default='pending')
    
    # Task details
    parameters = models.JSONField(default=dict, blank=True)
    result = models.JSONField(default=dict, blank=True)
    error_message = models.TextField(blank=True)
    
    # Timing information
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Processing Task"
        verbose_name_plural = "Processing Tasks"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.task_type} - {self.status} - {self.created_at}"


class Entity(models.Model):
    """
    Model to store entities extracted from documents (resumes and job descriptions).
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Entity information
    text = models.CharField(max_length=255)
    entity_type = models.CharField(max_length=100)  # SKILL, EDUCATION, EXPERIENCE, etc.
    
    # Source document
    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE, 
        null=True, blank=True, related_name='entities'
    )
    job_description = models.ForeignKey(
        JobDescription, on_delete=models.CASCADE, 
        null=True, blank=True, related_name='entities'
    )
    
    # Additional info
    confidence = models.FloatField(default=1.0)
    metadata = models.JSONField(default=dict, blank=True)
    
    # Creation timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Entity"
        verbose_name_plural = "Entities"
        ordering = ['entity_type', 'text']
    
    def __str__(self):
        return f"{self.entity_type}: {self.text}"


class DocumentChunk(models.Model):
    """
    Model to store chunks of text from documents for efficient embedding and retrieval.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Chunk text
    text = models.TextField()
    section_name = models.CharField(max_length=255, blank=True)
    
    # Source document
    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE, 
        null=True, blank=True, related_name='chunks'
    )
    job_description = models.ForeignKey(
        JobDescription, on_delete=models.CASCADE, 
        null=True, blank=True, related_name='chunks'
    )
    
    # Position in document
    chunk_index = models.PositiveIntegerField(default=0)
    
    # Embedding information
    embedding_generated = models.BooleanField(default=False)
    embedding_file = models.CharField(max_length=255, blank=True)
    
    # Creation timestamp
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Document Chunk"
        verbose_name_plural = "Document Chunks"
        ordering = ['resume', 'job_description', 'chunk_index']
    
    def __str__(self):
        if self.resume:
            return f"Resume chunk {self.chunk_index} from {self.resume.filename}"
        elif self.job_description:
            return f"Job Description chunk {self.chunk_index} from {self.job_description.filename}"
        return f"Document chunk {self.id}"