from django.urls import path
from apps.analysis import views

app_name = 'analysis'

urlpatterns = [
    # Candidate analysis routes
    path('', views.CandidateAnalysisListView.as_view(), name='candidate_list'),
    path('create/', views.CandidateAnalysisCreateView.as_view(), name='candidate_create'),
    path('<uuid:pk>/', views.CandidateAnalysisDetailView.as_view(), name='candidate_detail'),
    path('<uuid:pk>/update/', views.CandidateAnalysisUpdateView.as_view(), name='candidate_update'),
    path('<uuid:pk>/delete/', views.CandidateAnalysisDeleteView.as_view(), name='candidate_delete'),
    
    # AI-assisted analysis
    path('<uuid:pk>/generate-insights/', views.GenerateInsightsView.as_view(), name='generate_insights'),
    path('<uuid:pk>/feedback/', views.AIFeedbackView.as_view(), name='ai_feedback'),
    path('ask-ai/', views.AskAIView.as_view(), name='ask_ai'),
    
    # Question routes
    path('question/create/<uuid:analysis_id>/', views.CandidateQuestionCreateView.as_view(), name='question_create'),
    path('question/<uuid:pk>/', views.CandidateQuestionDetailView.as_view(), name='question_detail'),
    path('question/<uuid:pk>/update/', views.CandidateQuestionUpdateView.as_view(), name='question_update'),
    path('question/<uuid:pk>/delete/', views.CandidateQuestionDeleteView.as_view(), name='question_delete'),
    
    # Note routes
    path('note/create/<uuid:analysis_id>/', views.CandidateNoteCreateView.as_view(), name='note_create'),
    path('note/<uuid:pk>/', views.CandidateNoteDetailView.as_view(), name='note_detail'),
    path('note/<uuid:pk>/update/', views.CandidateNoteUpdateView.as_view(), name='note_update'),
    path('note/<uuid:pk>/delete/', views.CandidateNoteDeleteView.as_view(), name='note_delete'),
    
    # Export routes
    path('<uuid:pk>/export/', views.ExportAnalysisView.as_view(), name='export_analysis'),
]