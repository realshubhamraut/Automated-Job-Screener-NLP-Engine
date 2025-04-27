import uuid
from django.db import models
from django.conf import settings


class AppSetting(models.Model):
    """
    Model to store application settings and configurations.
    """
    key = models.CharField(max_length=100, unique=True)
    value = models.TextField()
    description = models.TextField(blank=True)
    default_value = models.TextField(blank=True)
    
    # Setting type for validation and UI rendering
    TYPE_CHOICES = (
        ('string', 'String'),
        ('number', 'Number'),
        ('boolean', 'Boolean'),
        ('json', 'JSON'),
        ('list', 'List'),
    )
    setting_type = models.CharField(max_length=20, choices=TYPE_CHOICES, default='string')
    
    # Setting group for organization
    SETTING_GROUPS = (
        ('general', 'General'),
        ('matching', 'Matching Engine'),
        ('document', 'Document Processing'),
        ('interview', 'Interview'),
        ('analytics', 'Analytics'),
        ('ui', 'User Interface'),
        ('security', 'Security'),
        ('api', 'API'),
    )
    setting_group = models.CharField(max_length=20, choices=SETTING_GROUPS, default='general')
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='updated_settings'
    )
    
    class Meta:
        verbose_name = "Application Setting"
        verbose_name_plural = "Application Settings"
    
    def __str__(self):
        return f"{self.key} ({self.setting_group})"


class SystemLog(models.Model):
    """
    Model to store system logs.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Log level
    LEVEL_CHOICES = (
        ('INFO', 'Information'),
        ('WARNING', 'Warning'),
        ('ERROR', 'Error'),
        ('DEBUG', 'Debug'),
        ('CRITICAL', 'Critical'),
    )
    level = models.CharField(max_length=10, choices=LEVEL_CHOICES, default='INFO')
    
    # Log content
    message = models.TextField()
    source = models.CharField(max_length=255, blank=True)  # Component that generated the log
    details = models.JSONField(default=dict, blank=True)  # Additional structured data
    
    # Metadata
    timestamp = models.DateTimeField(auto_now_add=True)
    user = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='system_logs'
    )
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    
    class Meta:
        verbose_name = "System Log"
        verbose_name_plural = "System Logs"
        ordering = ['-timestamp']
    
    def __str__(self):
        return f"{self.level}: {self.message[:50]}... ({self.timestamp.strftime('%Y-%m-%d %H:%M')})"


class ScheduledTask(models.Model):
    """
    Model to store scheduled tasks and track their execution.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    
    # Task definition
    task_path = models.CharField(max_length=255, help_text="Import path to the task function")
    arguments = models.JSONField(default=dict, blank=True)
    
    # Schedule
    FREQUENCY_CHOICES = (
        ('once', 'One-time'),
        ('hourly', 'Hourly'),
        ('daily', 'Daily'),
        ('weekly', 'Weekly'),
        ('monthly', 'Monthly'),
        ('custom', 'Custom'),
    )
    frequency = models.CharField(max_length=20, choices=FREQUENCY_CHOICES, default='once')
    cron_expression = models.CharField(max_length=100, blank=True, help_text="Cron expression for custom schedules")
    next_run = models.DateTimeField(null=True, blank=True)
    
    # Status
    is_active = models.BooleanField(default=True)
    last_run = models.DateTimeField(null=True, blank=True)
    last_run_status = models.CharField(max_length=20, blank=True)
    last_run_result = models.JSONField(default=dict, blank=True)
    
    # Metadata
    created_at = models.DateTimeField(auto_now_add=True)
    created_by = models.ForeignKey(
        settings.AUTH_USER_MODEL, on_delete=models.SET_NULL,
        null=True, blank=True, related_name='created_tasks'
    )
    
    class Meta:
        verbose_name = "Scheduled Task"
        verbose_name_plural = "Scheduled Tasks"
    
    def __str__(self):
        return self.name