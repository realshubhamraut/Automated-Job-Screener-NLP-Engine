from django.apps import AppConfig


class AnalysisConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.analysis'
    verbose_name = 'Candidate Analysis'
    
    def ready(self):
        """
        Import signals when the app is ready.
        """
        import apps.analysis.signals  # noqa