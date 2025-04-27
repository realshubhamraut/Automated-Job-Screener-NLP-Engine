"""
URL Configuration for AutomatedJobScreener project.
"""
from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('apps.core.urls')),
    path('accounts/', include('apps.accounts.urls')),
    path('documents/', include('apps.document_processor.urls')),
    path('match/', include('apps.matching_engine.urls')),
    path('analytics/', include('apps.analytics.urls')),
    path('interview/', include('apps.interview.urls')),
    path('analysis/', include('apps.analysis.urls')),
]

# Add static and media URLs in development
if settings.DEBUG:
    urlpatterns += static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)