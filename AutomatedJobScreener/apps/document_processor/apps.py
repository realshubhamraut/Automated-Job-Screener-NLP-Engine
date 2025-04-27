from django.apps import AppConfig


class DocumentProcessorConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'apps.document_processor'
    verbose_name = 'Document Processor'
    
    def ready(self):
        """
        Import signals when the app is ready.
        """
        import apps.document_processor.signals  