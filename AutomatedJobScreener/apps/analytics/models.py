import uuid
from django.db import models
from django.conf import settings
from apps.document_processor.models import Resume, JobDescription
from apps.matching_engine.models import MatchResult
from django.utils import timezone


class SkillMetric(models.Model):
    """
    Model to store metrics about skills across resumes and job descriptions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    skill_name = models.CharField(max_length=100)
    
    # Skill occurrences
    resume_count = models.PositiveIntegerField(default=0)  # Number of resumes with this skill
    job_count = models.PositiveIntegerField(default=0)  # Number of jobs requiring this skill
    
    # Normalized metrics (percentages)
    resume_percentage = models.FloatField(default=0.0)
    job_percentage = models.FloatField(default=0.0)
    
    # Metadata
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Skill Metric"
        verbose_name_plural = "Skill Metrics"
        ordering = ['-job_count', '-resume_count']
    
    def __str__(self):
        return f"{self.skill_name} (Jobs: {self.job_count}, Resumes: {self.resume_count})"


class SkillGapAnalysis(models.Model):
    """
    Model to store skill gap analysis results.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    
    # Analysis data
    total_resumes = models.PositiveIntegerField(default=0)
    total_jobs = models.PositiveIntegerField(default=0)
    job_skills_count = models.PositiveIntegerField(default=0)
    resume_skills_count = models.PositiveIntegerField(default=0)
    common_skills_count = models.PositiveIntegerField(default=0)
    skill_gaps_count = models.PositiveIntegerField(default=0)
    
    # Detailed data
    job_skills = models.JSONField(default=list)
    resume_skills = models.JSONField(default=list)
    common_skills = models.JSONField(default=list)
    skill_gaps = models.JSONField(default=list)
    extra_skills = models.JSONField(default=list)  # Skills in resumes but not in jobs
    
    # Optional filters
    job_descriptions = models.ManyToManyField(
        JobDescription, related_name='skill_gap_analyses', blank=True
    )
    resumes = models.ManyToManyField(
        Resume, related_name='skill_gap_analyses', blank=True
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='skill_gap_analyses'
    )
    
    class Meta:
        verbose_name = "Skill Gap Analysis"
        verbose_name_plural = "Skill Gap Analyses"
    
    def __str__(self):
        return self.name


class AnalyticsDashboard(models.Model):
    """
    Model to store dashboard configurations and cached analytics data.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Dashboard configuration
    config = models.JSONField(default=dict)
    
    # Cached analytics data
    resume_metrics = models.JSONField(default=dict)
    job_metrics = models.JSONField(default=dict)
    match_metrics = models.JSONField(default=dict)
    skill_metrics = models.JSONField(default=dict)
    
    # Optional filters for this dashboard
    job_descriptions = models.ManyToManyField(
        JobDescription, related_name='dashboards', blank=True
    )
    resumes = models.ManyToManyField(
        Resume, related_name='dashboards', blank=True
    )
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='dashboards'
    )
    
    class Meta:
        verbose_name = "Analytics Dashboard"
        verbose_name_plural = "Analytics Dashboards"
    
    def __str__(self):
        return self.name


class ResumeAnalysis(models.Model):
    """
    Model to store detailed analysis of a resume.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    resume = models.OneToOneField(
        Resume, on_delete=models.CASCADE, related_name='detailed_analysis'
    )
    
    # Analysis data
    word_count = models.PositiveIntegerField(default=0)
    sentence_count = models.PositiveIntegerField(default=0)
    avg_sentence_length = models.FloatField(default=0.0)
    
    # Extracted entities counts
    skill_count = models.PositiveIntegerField(default=0)
    education_count = models.PositiveIntegerField(default=0)
    experience_count = models.PositiveIntegerField(default=0)
    
    # Analysis results
    top_skills = models.JSONField(default=list)
    entity_distribution = models.JSONField(default=dict)
    content_analysis = models.JSONField(default=dict)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Resume Analysis"
        verbose_name_plural = "Resume Analyses"
    
    def __str__(self):
        return f"Analysis of {self.resume.filename}"


class JobAnalysis(models.Model):
    """
    Model to store detailed analysis of a job description.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    job_description = models.OneToOneField(
        JobDescription, on_delete=models.CASCADE, related_name='detailed_analysis'
    )
    
    # Analysis data
    word_count = models.PositiveIntegerField(default=0)
    sentence_count = models.PositiveIntegerField(default=0)
    avg_sentence_length = models.FloatField(default=0.0)
    
    # Extracted entities counts
    required_skill_count = models.PositiveIntegerField(default=0)
    preferred_skill_count = models.PositiveIntegerField(default=0)
    
    # Analysis results
    top_skills = models.JSONField(default=list)
    keyword_frequency = models.JSONField(default=dict)
    section_analysis = models.JSONField(default=dict)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Job Analysis"
        verbose_name_plural = "Job Analyses"
    
    def __str__(self):
        return f"Analysis of {self.job_description.filename}"


class AnalyticsSnapshot(models.Model):
    """
    Model to store periodic snapshots of system analytics.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Snapshot metadata
    title = models.CharField(max_length=255)
    snapshot_date = models.DateTimeField(default=timezone.now)
    
    # Resume statistics
    resume_count = models.IntegerField(default=0)
    avg_resume_length = models.IntegerField(default=0)  # In characters
    avg_entities_per_resume = models.FloatField(default=0.0)
    
    # Job description statistics
    job_count = models.IntegerField(default=0)
    avg_job_length = models.IntegerField(default=0)  # In characters
    avg_entities_per_job = models.FloatField(default=0.0)
    
    # Match statistics
    match_count = models.IntegerField(default=0)
    avg_match_score = models.FloatField(default=0.0)
    median_match_score = models.FloatField(default=0.0)
    strong_match_count = models.IntegerField(default=0)  # Matches >= 0.7
    
    # Skill gap statistics
    total_job_skills = models.IntegerField(default=0)
    total_resume_skills = models.IntegerField(default=0)
    common_skills_count = models.IntegerField(default=0)
    skill_gap_count = models.IntegerField(default=0)
    
    # Raw data for charts
    resume_length_distribution = models.JSONField(default=list, blank=True)
    job_length_distribution = models.JSONField(default=list, blank=True)
    match_score_distribution = models.JSONField(default=list, blank=True)
    top_skills = models.JSONField(default=list, blank=True)
    top_required_skills = models.JSONField(default=list, blank=True)
    skill_gap_details = models.JSONField(default=dict, blank=True)
    
    # User who generated this snapshot
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='analytics_snapshots'
    )
    
    class Meta:
        verbose_name = "Analytics Snapshot"
        verbose_name_plural = "Analytics Snapshots"
        ordering = ['-snapshot_date']
    
    def __str__(self):
        return f"{self.title} ({self.snapshot_date.strftime('%Y-%m-%d')})"


class SkillTrend(models.Model):
    """
    Model to track skill trends over time.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # The skill being tracked
    skill_name = models.CharField(max_length=255)
    
    # Trend data
    trend_date = models.DateField(default=timezone.now)
    demand_count = models.IntegerField(default=0)  # Count in job descriptions
    supply_count = models.IntegerField(default=0)  # Count in resumes
    gap_score = models.FloatField(default=0.0)  # Demand - Supply (normalized)
    
    class Meta:
        verbose_name = "Skill Trend"
        verbose_name_plural = "Skill Trends"
        unique_together = ('skill_name', 'trend_date')
        ordering = ['-trend_date', 'skill_name']
    
    def __str__(self):
        return f"{self.skill_name} trend on {self.trend_date}"


class Dashboard(models.Model):
    """
    Model to store user dashboard configurations.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Basic information
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Configuration
    layout = models.JSONField(default=dict)
    chart_settings = models.JSONField(default=dict)
    filter_settings = models.JSONField(default=dict)
    
    # Dashboard access
    is_public = models.BooleanField(default=False)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='dashboards'
    )
    shared_with = models.ManyToManyField(
        settings.AUTH_USER_MODEL,
        related_name='shared_dashboards',
        blank=True
    )
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Dashboard"
        verbose_name_plural = "Dashboards"
        ordering = ['-updated_at']
    
    def __str__(self):
        return f"{self.title} ({self.user.username})"


class Report(models.Model):
    """
    Model to store generated analytics reports.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Report information
    title = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Report content
    content = models.JSONField(default=dict)
    
    # Report type and configuration
    REPORT_TYPE_CHOICES = (
        ('resume_analytics', 'Resume Analytics'),
        ('job_analytics', 'Job Description Analytics'),
        ('match_analytics', 'Match Analytics'),
        ('skill_gap', 'Skill Gap Analysis'),
        ('custom', 'Custom Report'),
    )
    report_type = models.CharField(max_length=30, choices=REPORT_TYPE_CHOICES)
    
    # Generated outputs
    pdf_file = models.FileField(upload_to='reports/pdf/', blank=True, null=True)
    excel_file = models.FileField(upload_to='reports/excel/', blank=True, null=True)
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='generated_reports'
    )
    
    class Meta:
        verbose_name = "Report"
        verbose_name_plural = "Reports"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} ({self.report_type})"


class CustomMetric(models.Model):
    """
    Model to store custom metrics defined by users.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Metric information
    name = models.CharField(max_length=255)
    description = models.TextField(blank=True)
    
    # Metric calculation
    calculation_formula = models.TextField(help_text="Python code or formula to calculate this metric")
    
    # Input sources
    input_fields = models.JSONField(default=list)
    
    # Metric settings
    display_format = models.CharField(max_length=50, default="{value:.2f}")
    display_unit = models.CharField(max_length=50, blank=True)
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.CASCADE,
        related_name='custom_metrics'
    )
    
    class Meta:
        verbose_name = "Custom Metric"
        verbose_name_plural = "Custom Metrics"
        ordering = ['name']
    
    def __str__(self):
        return self.name