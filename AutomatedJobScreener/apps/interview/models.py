import os
import uuid
from django.db import models
from django.conf import settings
from apps.document_processor.models import Resume
from apps.matching_engine.models import MatchResult, ShortlistEntry


class InterviewSession(models.Model):
    """
    Model for tracking complete interview sessions with candidates.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Session context
    title = models.CharField(max_length=255)
    match_result = models.ForeignKey(
        MatchResult, on_delete=models.CASCADE,
        null=True, blank=True, related_name='interview_sessions'
    )
    resume = models.ForeignKey(
        Resume, on_delete=models.CASCADE,
        related_name='interview_sessions'
    )
    
    # Session management
    STATUS_CHOICES = (
        ('scheduled', 'Scheduled'),
        ('in_progress', 'In Progress'),
        ('completed', 'Completed'),
        ('cancelled', 'Cancelled'),
    )
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='scheduled')
    scheduled_date = models.DateTimeField(null=True, blank=True)
    completion_date = models.DateTimeField(null=True, blank=True)
    
    # Session configuration
    session_type = models.CharField(max_length=50, default='standard')  # standard, technical, behavioral
    session_config = models.JSONField(default=dict, blank=True)
    
    # Overall results
    overall_score = models.FloatField(null=True, blank=True)
    overall_feedback = models.TextField(blank=True)
    interviewer_notes = models.TextField(blank=True)
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='created_sessions'
    )
    
    class Meta:
        verbose_name = "Interview Session"
        verbose_name_plural = "Interview Sessions"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"{self.title} for {self.resume.candidate_name or self.resume.filename}"
    
    @property
    def question_count(self):
        """Count the number of questions in this session."""
        return self.session_questions.count()
    
    @property
    def completed_response_count(self):
        """Count the number of questions that have been answered."""
        return CandidateResponse.objects.filter(
            question__in=self.session_questions.all()
        ).count()
    
    @property
    def completion_percentage(self):
        """Calculate the percentage of the interview that has been completed."""
        if self.question_count == 0:
            return 0
        return (self.completed_response_count / self.question_count) * 100


class InterviewQuestion(models.Model):
    """
    Model for storing generated interview questions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Question content
    question = models.TextField()
    question_type = models.CharField(max_length=50)  # Technical, Behavioral, Experience, etc.
    difficulty = models.CharField(max_length=20)  # Basic, Intermediate, Advanced, Expert
    purpose = models.TextField(blank=True)
    good_answer_criteria = models.TextField(blank=True)
    
    # Context
    match_result = models.ForeignKey(
        MatchResult, on_delete=models.CASCADE,
        null=True, blank=True, related_name='interview_questions'
    )
    shortlist_entry = models.ForeignKey(
        ShortlistEntry, on_delete=models.CASCADE,
        null=True, blank=True, related_name='interview_questions'
    )
    
    # Generation method
    GENERATION_METHOD_CHOICES = (
        ('rule_based', 'Rule Based'),
        ('bert', 'BERT Model'),
        ('gemini', 'Gemini AI'),
        ('custom', 'Custom'),
        ('manual', 'Manually Created'),
    )
    generation_method = models.CharField(
        max_length=20, 
        choices=GENERATION_METHOD_CHOICES,
        default='rule_based'
    )
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='created_questions'
    )
    
    class Meta:
        verbose_name = "Interview Question"
        verbose_name_plural = "Interview Questions"
        ordering = ['question_type', 'difficulty']
    
    def __str__(self):
        return f"{self.question_type} question ({self.difficulty}): {self.question[:50]}..."


class CandidateResponse(models.Model):
    """
    Model for storing candidate responses to interview questions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Related question
    question = models.ForeignKey(
        InterviewQuestion, on_delete=models.CASCADE,
        related_name='responses'
    )
    
    # Response content
    text_response = models.TextField(blank=True)
    audio_file = models.FileField(upload_to='interview_responses/', blank=True, null=True)
    code_response = models.TextField(blank=True)
    
    # Response evaluation
    evaluation_score = models.FloatField(null=True, blank=True)
    evaluation_details = models.JSONField(default=dict, blank=True)
    feedback = models.TextField(blank=True)
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Candidate Response"
        verbose_name_plural = "Candidate Responses"
        ordering = ['-created_at']
    
    def __str__(self):
        return f"Response to: {self.question.question[:30]}... (Score: {self.evaluation_score or 'N/A'})"


class SessionQuestion(models.Model):
    """
    Model for linking questions to interview sessions and tracking their order.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    session = models.ForeignKey(
        InterviewSession, on_delete=models.CASCADE,
        related_name='session_questions'
    )
    question = models.ForeignKey(
        InterviewQuestion, on_delete=models.CASCADE,
        related_name='session_occurrences'
    )
    
    # Question order within the session
    order = models.PositiveIntegerField(default=0)
    
    # Status tracking
    is_answered = models.BooleanField(default=False)
    response = models.ForeignKey(
        CandidateResponse, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='session_question'
    )
    
    class Meta:
        verbose_name = "Session Question"
        verbose_name_plural = "Session Questions"
        ordering = ['session', 'order']
        unique_together = ('session', 'question')
    
    def __str__(self):
        return f"Q{self.order+1} in {self.session.title}"


class CodingChallenge(models.Model):
    """
    Model for storing coding challenges associated with interview questions.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Associated interview question
    question = models.OneToOneField(
        InterviewQuestion, on_delete=models.CASCADE,
        related_name='coding_challenge'
    )
    
    # Challenge content
    problem_statement = models.TextField()
    input_format = models.TextField()
    output_format = models.TextField()
    constraints = models.TextField()
    examples = models.JSONField(default=list)
    starter_code = models.TextField(blank=True)
    solution = models.TextField()
    
    # Test cases for automated evaluation
    test_cases = models.JSONField(default=list)
    
    # Management fields
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        verbose_name = "Coding Challenge"
        verbose_name_plural = "Coding Challenges"
    
    def __str__(self):
        return f"Coding challenge for: {self.question.question[:50]}..."


class VoiceResponse(models.Model):
    """
    Model to store voice responses from candidates.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    question = models.ForeignKey(
        InterviewQuestion, on_delete=models.CASCADE, related_name='voice_responses'
    )
    
    # Voice data
    audio_file = models.FileField(upload_to='voice_recordings/', blank=True, null=True)
    transcription = models.TextField(blank=True)
    duration_seconds = models.FloatField(default=0.0)
    
    # Analysis results
    analysis = models.JSONField(default=dict)
    
    # Metrics (extracted from analysis for easy querying)
    relevance_score = models.FloatField(default=0.0)
    clarity_score = models.FloatField(default=0.0)
    technical_accuracy = models.FloatField(default=0.0)
    confidence_score = models.FloatField(default=0.0)
    overall_score = models.FloatField(default=0.0)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Voice Response"
        verbose_name_plural = "Voice Responses"
    
    def __str__(self):
        return f"Response to {self.question}"
    
    def save_audio(self, audio_data):
        """Save audio data to a file"""
        filename = f"voice_{self.id}.wav"
        self.audio_file.save(filename, audio_data)


class CodeSubmission(models.Model):
    """
    Model to store code submissions for challenges.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    challenge = models.ForeignKey(
        CodingChallenge, on_delete=models.CASCADE, related_name='submissions'
    )
    
    # Submission data
    submitted_code = models.TextField()
    language = models.CharField(max_length=50, default='python')
    
    # Evaluation results
    test_results = models.JSONField(default=list)
    
    # AI evaluation scores
    correctness_score = models.FloatField(default=0.0)
    efficiency_score = models.FloatField(default=0.0)
    code_quality_score = models.FloatField(default=0.0)
    overall_score = models.FloatField(default=0.0)
    
    # AI evaluation feedback
    strengths = models.JSONField(default=list)
    areas_for_improvement = models.JSONField(default=list)
    overall_evaluation = models.TextField(blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    
    class Meta:
        verbose_name = "Code Submission"
        verbose_name_plural = "Code Submissions"
    
    def __str__(self):
        return f"Submission for {self.challenge}"


class CandidateEmail(models.Model):
    """
    Model to store emails sent to candidates.
    """
    EMAIL_TYPE_CHOICES = (
        ('feedback', 'Interview Feedback'),
        ('next_steps', 'Next Steps'),
        ('custom', 'Custom Email'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    session = models.ForeignKey(
        InterviewSession, on_delete=models.CASCADE, related_name='emails'
    )
    
    # Email details
    email_type = models.CharField(max_length=20, choices=EMAIL_TYPE_CHOICES, default='feedback')
    recipient_email = models.EmailField()
    subject = models.CharField(max_length=255)
    html_content = models.TextField()
    
    # Status
    is_sent = models.BooleanField(default=False)
    sent_at = models.DateTimeField(null=True, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, 
        null=True, blank=True, related_name='candidate_emails'
    )
    
    class Meta:
        verbose_name = "Candidate Email"
        verbose_name_plural = "Candidate Emails"
    
    def __str__(self):
        return f"Email to {self.recipient_email} ({self.email_type})"