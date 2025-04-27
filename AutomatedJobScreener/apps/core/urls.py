from django.urls import path
from apps.core import views

app_name = 'core'

urlpatterns = [
    path('', views.HomeView.as_view(), name='home'),
    path('dashboard/', views.DashboardView.as_view(), name='dashboard'),
    path('settings/', views.SettingsView.as_view(), name='settings'),
    path('help/', views.HelpView.as_view(), name='help'),
    path('about/', views.AboutView.as_view(), name='about'),
]