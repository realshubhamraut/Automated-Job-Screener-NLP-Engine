"""
WebSocket URL routing configuration for the API.
"""
from django.urls import path
from api.consumers import (
    InterviewConsumer,
    VoiceRecordingConsumer,
    CodeAssessmentConsumer,
)

websocket_urlpatterns = [
    path('ws/interview/<str:interview_id>/', InterviewConsumer.as_asgi()),
    path('ws/voice/<str:session_id>/', VoiceRecordingConsumer.as_asgi()),
    path('ws/code/<str:assessment_id>/', CodeAssessmentConsumer.as_asgi()),
]