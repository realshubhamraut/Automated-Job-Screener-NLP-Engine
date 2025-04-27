from django.shortcuts import render, redirect
from django.views.generic import TemplateView, View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.contrib import messages

from apps.core.models import AppSetting, SystemLog
from apps.document_processor.models import Resume, JobDescription
from apps.matching_engine.models import MatchResult


class HomeView(TemplateView):
    """
    Home page view. This replaces the Streamlit app's main page.
    """
    template_name = 'core/home.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add counts for dashboard summary
        if self.request.user.is_authenticated:
            context['resume_count'] = Resume.objects.filter(user=self.request.user).count()
            context['job_count'] = JobDescription.objects.filter(user=self.request.user).count()
            context['match_count'] = MatchResult.objects.filter(created_by=self.request.user).count()
        else:
            context['resume_count'] = Resume.objects.count()
            context['job_count'] = JobDescription.objects.count()
            context['match_count'] = MatchResult.objects.count()
            
        return context


class DashboardView(LoginRequiredMixin, TemplateView):
    """
    Main dashboard view for authenticated users.
    """
    template_name = 'core/dashboard.html'
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        
        # Add recent items and statistics
        context['recent_resumes'] = Resume.objects.filter(user=self.request.user).order_by('-uploaded_at')[:5]
        context['recent_jobs'] = JobDescription.objects.filter(user=self.request.user).order_by('-uploaded_at')[:5]
        context['recent_matches'] = MatchResult.objects.filter(created_by=self.request.user).order_by('-created_at')[:5]
        
        # Get system logs for admin users
        if self.request.user.is_staff:
            context['system_logs'] = SystemLog.objects.order_by('-timestamp')[:10]
        
        return context


class SettingsView(LoginRequiredMixin, View):
    """
    View for app settings.
    """
    template_name = 'core/settings.html'
    
    def get(self, request):
        context = {
            'settings': AppSetting.objects.all(),
        }
        return render(request, self.template_name, context)
    
    def post(self, request):
        # Process settings form submission
        for key, value in request.POST.items():
            if key.startswith('setting_'):
                setting_key = key.replace('setting_', '')
                setting, created = AppSetting.objects.update_or_create(
                    key=setting_key,
                    defaults={'value': value, 'updated_by': request.user}
                )
        
        messages.success(request, 'Settings updated successfully.')
        return redirect('core:settings')


class HelpView(TemplateView):
    """
    Help page view.
    """
    template_name = 'core/help.html'


class AboutView(TemplateView):
    """
    About page view.
    """
    template_name = 'core/about.html'