from django.db import models
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


class UserProfile(models.Model):
    """
    Extended user profile model to store additional user information.
    """
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    
    # Organization info
    organization = models.CharField(max_length=255, blank=True)
    job_title = models.CharField(max_length=255, blank=True)
    department = models.CharField(max_length=255, blank=True)
    
    # Contact info
    phone_number = models.CharField(max_length=20, blank=True)
    
    # Profile settings
    avatar = models.ImageField(upload_to='avatars/', null=True, blank=True)
    theme_preference = models.CharField(max_length=20, default='light')
    
    # API keys and integration settings
    api_keys = models.JSONField(default=dict, blank=True)
    integration_settings = models.JSONField(default=dict, blank=True)
    
    # Usage statistics
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    documents_processed = models.PositiveIntegerField(default=0)
    matches_run = models.PositiveIntegerField(default=0)
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.user.username}'s Profile"


# Create user profile automatically when a new user is created
@receiver(post_save, sender=User)
def create_user_profile(sender, instance, created, **kwargs):
    if created:
        UserProfile.objects.create(user=instance)


@receiver(post_save, sender=User)
def save_user_profile(sender, instance, **kwargs):
    instance.profile.save()