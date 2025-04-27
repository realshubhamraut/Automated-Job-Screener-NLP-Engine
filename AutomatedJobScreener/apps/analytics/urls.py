from django.urls import path
from apps.analytics import views

app_name = 'analytics'

urlpatterns = [
    # Dashboard routes
    path('', views.DashboardView.as_view(), name='dashboard'),
    path('resume-analytics/', views.ResumeAnalyticsView.as_view(), name='resume_analytics'),
    path('job-analytics/', views.JobAnalyticsView.as_view(), name='job_analytics'),
    path('match-analytics/', views.MatchAnalyticsView.as_view(), name='match_analytics'),
    path('skill-gap-analysis/', views.SkillGapAnalysisView.as_view(), name='skill_gap_analysis'),
    
    # Detailed analytics routes
    path('resume-analytics/<uuid:pk>/', views.ResumeAnalysisDetailView.as_view(), name='resume_analysis_detail'),
    path('job-analytics/<uuid:pk>/', views.JobAnalysisDetailView.as_view(), name='job_analysis_detail'),
    
    # Dashboards management
    path('dashboards/', views.DashboardListView.as_view(), name='dashboard_list'),
    path('dashboards/create/', views.DashboardCreateView.as_view(), name='dashboard_create'),
    path('dashboards/<uuid:pk>/', views.DashboardDetailView.as_view(), name='dashboard_detail'),
    path('dashboards/<uuid:pk>/update/', views.DashboardUpdateView.as_view(), name='dashboard_update'),
    path('dashboards/<uuid:pk>/delete/', views.DashboardDeleteView.as_view(), name='dashboard_delete'),
    
    # Export routes
    path('export/skill-gap-analysis/', views.ExportSkillGapAnalysisView.as_view(), name='export_skill_gap_analysis'),
    path('export/resume-analytics/', views.ExportResumeAnalyticsView.as_view(), name='export_resume_analytics'),
    path('export/job-analytics/', views.ExportJobAnalyticsView.as_view(), name='export_job_analytics'),
    path('export/dashboard/<uuid:pk>/', views.ExportDashboardView.as_view(), name='export_dashboard'),
]