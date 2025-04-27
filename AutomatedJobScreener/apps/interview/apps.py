from django.apps import AppConfig


class InterviewConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.interview'
    verbose_name = 'Interview'
    
    def ready(self):
        """
        Import signals when the app is ready.
        """
        import apps.interview.signals  # noqa