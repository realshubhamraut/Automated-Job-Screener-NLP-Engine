from django.urls import path
from apps.matching_engine import views

app_name = 'matching_engine'

urlpatterns = [
    # Match routes
    path('', views.MatchHomeView.as_view(), name='match_home'),
    path('create/', views.MatchCreateView.as_view(), name='match_create'),
    path('results/', views.MatchResultListView.as_view(), name='match_results'),
    path('results/<uuid:pk>/', views.MatchResultDetailView.as_view(), name='match_detail'),
    path('results/<uuid:pk>/delete/', views.MatchResultDeleteView.as_view(), name='match_delete'),
    path('batch/', views.MatchBatchListView.as_view(), name='match_batch_list'),
    path('batch/<uuid:pk>/', views.MatchBatchDetailView.as_view(), name='match_batch_detail'),
    path('batch/create/', views.MatchBatchCreateView.as_view(), name='match_batch_create'),
    
    # Shortlist routes
    path('shortlists/', views.ShortlistListView.as_view(), name='shortlist_list'),
    path('shortlists/create/', views.ShortlistCreateView.as_view(), name='shortlist_create'),
    path('shortlists/<uuid:pk>/', views.ShortlistDetailView.as_view(), name='shortlist_detail'),
    path('shortlists/<uuid:pk>/update/', views.ShortlistUpdateView.as_view(), name='shortlist_update'),
    path('shortlists/<uuid:pk>/delete/', views.ShortlistDeleteView.as_view(), name='shortlist_delete'),
    
    # Match settings
    path('settings/', views.MatchSettingsView.as_view(), name='match_settings'),
]