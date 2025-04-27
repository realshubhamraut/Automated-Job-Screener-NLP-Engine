import uuid
from django.db import models
from django.conf import settings
from apps.document_processor.models import Resume, JobDescription
from apps.matching_engine.models import MatchResult


class CandidateAnalysis(models.Model):
    """
    Model to store detailed candidate analysis information.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    resume = models.OneToOneField(
        Resume, on_delete=models.CASCADE, related_name='candidate_analysis'
    )
    job_description = models.ForeignKey(
        JobDescription, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='candidate_analyses'
    )
    match_result = models.ForeignKey(
        MatchResult, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='candidate_analyses'
    )
    
    # Contact information extracted from resume
    name = models.CharField(max_length=255, blank=True)
    email = models.EmailField(blank=True)
    phone = models.CharField(max_length=20, blank=True)
    location = models.CharField(max_length=255, blank=True)
    
    # Professional details
    job_title = models.CharField(max_length=255, blank=True)
    experience_years = models.PositiveIntegerField(default=0)
    summary = models.TextField(blank=True)
    
    # Analysis data
    skills = models.JSONField(default=list)  # List of skills from resume
    common_skills = models.JSONField(default=list)  # Skills in both resume and job
    missing_skills = models.JSONField(default=list)  # Skills in job but not in resume
    skill_match_percentage = models.FloatField(default=0.0)
    
    # AI-generated insights
    insights = models.JSONField(default=dict)
    ai_feedback = models.TextField(blank=True)
    
    # Status
    is_shortlisted = models.BooleanField(default=False)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='candidate_analyses'
    )
    
    class Meta:
        verbose_name = "Candidate Analysis"
        verbose_name_plural = "Candidate Analyses"
        unique_together = ('resume', 'job_description')
    
    def __str__(self):
        return f"Analysis for {self.name or 'unknown candidate'}"


class CandidateQuestion(models.Model):
    """
    Model to store questions asked about a candidate.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    candidate_analysis = models.ForeignKey(
        CandidateAnalysis, on_delete=models.CASCADE, related_name='questions'
    )
    
    # Question and answer
    question = models.TextField()
    answer = models.TextField()
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='candidate_questions'
    )
    
    class Meta:
        verbose_name = "Candidate Question"
        verbose_name_plural = "Candidate Questions"
    
    def __str__(self):
        return f"Q: {self.question[:50]}..."


class CandidateNote(models.Model):
    """
    Model to store notes about a candidate.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    candidate_analysis = models.ForeignKey(
        CandidateAnalysis, on_delete=models.CASCADE, related_name='notes'
    )
    
    # Note content
    title = models.CharField(max_length=255)
    content = models.TextField()
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='candidate_notes'
    )
    
    class Meta:
        verbose_name = "Candidate Note"
        verbose_name_plural = "Candidate Notes"
    
    def __str__(self):
        return self.title