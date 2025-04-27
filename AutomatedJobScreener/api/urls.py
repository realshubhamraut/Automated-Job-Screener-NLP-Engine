"""
URL Configuration for API endpoints.
"""
from django.urls import path, include
from rest_framework.routers import DefaultRouter
from api.views import (
    ResumeViewSet,
    JobDescriptionViewSet,
    MatchResultViewSet,
    InterviewViewSet,
    VoiceRecordingViewSet,
    CodeAssessmentViewSet,
    AnalyticsViewSet,
)

# Create a router and register our viewsets
router = DefaultRouter()
router.register(r'resumes', ResumeViewSet, basename='resume')
router.register(r'jobs', JobDescriptionViewSet, basename='job-description')
router.register(r'matches', MatchResultViewSet, basename='match-result')
router.register(r'interviews', InterviewViewSet, basename='interview')
router.register(r'voice-recordings', VoiceRecordingViewSet, basename='voice-recording')
router.register(r'code-assessments', CodeAssessmentViewSet, basename='code-assessment')
router.register(r'analytics', AnalyticsViewSet, basename='analytics')

urlpatterns = [
    path('', include(router.urls)),
    path('auth/', include('rest_framework.urls', namespace='rest_framework')),
]