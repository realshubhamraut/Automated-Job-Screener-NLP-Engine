from django.urls import path
from apps.document_processor import views

app_name = 'document_processor'

urlpatterns = [
    # Resume routes
    path('resumes/', views.ResumeListView.as_view(), name='resume_list'),
    path('resumes/create/', views.ResumeCreateView.as_view(), name='resume_create'),
    path('resumes/<uuid:pk>/', views.ResumeDetailView.as_view(), name='resume_detail'),
    path('resumes/<uuid:pk>/update/', views.ResumeUpdateView.as_view(), name='resume_update'),
    path('resumes/<uuid:pk>/delete/', views.ResumeDeleteView.as_view(), name='resume_delete'),
    path('resumes/<uuid:pk>/process/', views.ResumeProcessView.as_view(), name='resume_process'),
    path('resumes/upload-batch/', views.ResumeBatchUploadView.as_view(), name='resume_batch_upload'),
    
    # Job description routes
    path('jobs/', views.JobDescriptionListView.as_view(), name='job_list'),
    path('jobs/create/', views.JobDescriptionCreateView.as_view(), name='job_create'),
    path('jobs/<uuid:pk>/', views.JobDescriptionDetailView.as_view(), name='job_detail'),
    path('jobs/<uuid:pk>/update/', views.JobDescriptionUpdateView.as_view(), name='job_update'),
    path('jobs/<uuid:pk>/delete/', views.JobDescriptionDeleteView.as_view(), name='job_delete'),
    path('jobs/<uuid:pk>/process/', views.JobDescriptionProcessView.as_view(), name='job_process'),
    path('jobs/upload-batch/', views.JobDescriptionBatchUploadView.as_view(), name='job_batch_upload'),
    
    # Entities and embeddings
    path('entities/', views.EntityListView.as_view(), name='entity_list'),
    path('entities/<uuid:pk>/', views.EntityDetailView.as_view(), name='entity_detail'),
]