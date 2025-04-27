from django.urls import path
from apps.interview import views

app_name = 'interview'

urlpatterns = [
    # Interview session routes
    path('', views.InterviewSessionListView.as_view(), name='session_list'),
    path('create/', views.InterviewSessionCreateView.as_view(), name='session_create'),
    path('generate/<uuid:match_id>/', views.GenerateQuestionsView.as_view(), name='generate_questions'),
    path('<uuid:pk>/', views.InterviewSessionDetailView.as_view(), name='session_detail'),
    path('<uuid:pk>/delete/', views.InterviewSessionDeleteView.as_view(), name='session_delete'),
    
    # Question routes
    path('questions/<uuid:session_id>/', views.QuestionListView.as_view(), name='question_list'),
    path('questions/<uuid:pk>/detail/', views.QuestionDetailView.as_view(), name='question_detail'),
    
    # Voice recording routes
    path('voice/record/<uuid:question_id>/', views.VoiceRecordingCreateView.as_view(), name='voice_record'),
    path('voice/<uuid:pk>/', views.VoiceRecordingDetailView.as_view(), name='voice_detail'),
    path('voice/<uuid:pk>/analyze/', views.VoiceRecordingAnalysisView.as_view(), name='voice_analyze'),
    
    # Code assessment routes
    path('code/create/<uuid:question_id>/', views.CodeChallengeCreateView.as_view(), name='code_create'),
    path('code/<uuid:pk>/', views.CodeChallengeDetailView.as_view(), name='code_detail'),
    path('code/<uuid:pk>/submit/', views.CodeSubmissionCreateView.as_view(), name='code_submit'),
    path('code/submission/<uuid:pk>/', views.CodeSubmissionDetailView.as_view(), name='submission_detail'),
    path('code/submission/<uuid:pk>/evaluate/', views.CodeSubmissionEvaluateView.as_view(), name='submission_evaluate'),
    
    # Email routes
    path('email/feedback/<uuid:session_id>/', views.SendFeedbackEmailView.as_view(), name='email_feedback'),
    path('email/next-steps/<uuid:session_id>/', views.SendNextStepsEmailView.as_view(), name='email_next_steps'),
    path('email/custom/<uuid:session_id>/', views.SendCustomEmailView.as_view(), name='email_custom'),
    
    # Dashboard and reporting
    path('dashboard/', views.InterviewDashboardView.as_view(), name='dashboard'),
    path('reports/', views.InterviewReportingView.as_view(), name='reporting'),
    path('export/<uuid:session_id>/', views.ExportInterviewDataView.as_view(), name='export'),
]