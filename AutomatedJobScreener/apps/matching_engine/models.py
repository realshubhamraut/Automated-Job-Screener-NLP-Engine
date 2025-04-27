import uuid
from django.db import models
from django.conf import settings
from apps.document_processor.models import Resume, JobDescription


class MatchResult(models.Model):
    """
    Model to store matching results between resumes and job descriptions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Matched documents
    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE,
        related_name='match_results'
    )
    job_description = models.ForeignKey(
        JobDescription, on_delete=models.CASCADE,
        related_name='match_results'
    )
    
    # Matching results
    score = models.FloatField(help_text="Overall match score between 0 and 1")
    semantic_similarity = models.FloatField(help_text="Semantic similarity score")
    keyword_similarity = models.FloatField(help_text="Keyword-based similarity score")
    
    # Result details
    details = models.JSONField(default=dict, blank=True)
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='created_matches'
    )
    
    class Meta:
        verbose_name = "Match Result"
        verbose_name_plural = "Match Results"
        unique_together = ('resume', 'job_description')
        ordering = ['-score']
    
    def __str__(self):
        return f"Match: {self.resume.filename} - {self.job_description.filename} ({self.score:.2f})"


class MatchBatch(models.Model):
    """
    Model to track batch matching operations.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Batch status
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Batch configuration
    configuration = models.JSONField(default=dict, blank=True)
    
    # Results summary
    total_combinations = models.IntegerField(default=0)
    processed_count = models.IntegerField(default=0)
    success_count = models.IntegerField(default=0)
    error_count = models.IntegerField(default=0)
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='created_match_batches'
    )
    
    class Meta:
        verbose_name = "Match Batch"
        verbose_name_plural = "Match Batches"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.name} ({self.status})"


class MatchBatchItem(models.Model):
    """
    Model to track individual items within a batch matching operation.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    batch = models.ForeignKey(
        MatchBatch, on_delete=models.CASCADE,
        related_name='batch_items'
    )
    
    # Items to match
    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE,
        related_name='batch_items'
    )
    job_description = models.ForeignKey(
        JobDescription, on_delete=models.CASCADE,
        related_name='batch_items'
    )
    
    # Processing status
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('processing', 'Processing'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    
    # Result reference (if successful)
    match_result = models.ForeignKey(
        MatchResult, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='batch_item'
    )
    
    # Error information (if failed)
    error_message = models.TextField(blank=True)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        verbose_name = "Match Batch Item"
        verbose_name_plural = "Match Batch Items"
    
    def __str__(self):
        return f"Batch item: {self.resume.filename} - {self.job_description.filename} ({self.status})"


class Shortlist(models.Model):
    """
    Model to store shortlisted candidates for specific job descriptions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Job description this shortlist is for
    job_description = models.ForeignKey(
        JobDescription, on_delete=models.CASCADE,
        related_name='shortlists'
    )
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='created_shortlists'
    )
    
    class Meta:
        verbose_name = "Shortlist"
        verbose_name_plural = "Shortlists"
    
    def __str__(self):
        return f"{self.name} for {self.job_description.filename}"


class ShortlistEntry(models.Model):
    """
    Model to store individual entries in a shortlist.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    shortlist = models.ForeignKey(
        Shortlist, on_delete=models.CASCADE,
        related_name='entries'
    )
    
    # Match result this entry is based on
    match_result = models.ForeignKey(
        MatchResult, on_delete=models.CASCADE,
        related_name='shortlist_entries'
    )
    
    # Candidate information (denormalized for convenience)
    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE,
        related_name='shortlist_entries'
    )
    
    # Additional fields for shortlist management
    notes = models.TextField(blank=True)
    rating = models.PositiveIntegerField(default=0, help_text="Rating from 0 to 5")
    
    # Status of the candidate in the hiring process
    STATUS_CHOICES = (
        ('new', 'New'),
        ('contacted', 'Contacted'),
        ('interview_scheduled', 'Interview Scheduled'),
        ('interviewed', 'Interviewed'),
        ('offer_extended', 'Offer Extended'),
        ('hired', 'Hired'),
        ('rejected', 'Rejected'),
        ('declined', 'Declined'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Shortlist Entry"
        verbose_name_plural = "Shortlist Entries"
        ordering = ['-match_result__score']
        unique_together = ('shortlist', 'resume')
    
    def __str__(self):
        return f"{self.resume.filename} - {self.status} ({self.match_result.score:.2f})"