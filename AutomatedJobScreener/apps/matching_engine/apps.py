from django.apps import AppConfig


class MatchingEngineConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.matching_engine'
    verbose_name = 'Matching Engine'
    
    def ready(self):
        """
        Import signals when the app is ready.
        """
        import apps.matching_engine.signals  # noqa