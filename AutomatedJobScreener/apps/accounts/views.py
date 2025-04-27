from django.shortcuts import render, redirect
from django.views.generic import View, CreateView, UpdateView, DetailView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

from apps.accounts.models import UserProfile


class RegisterView(CreateView):
    """
    View for user registration.
    """
    template_name = 'accounts/register.html'
    form_class = UserCreationForm
    success_url = reverse_lazy('accounts:login')
    
    def form_valid(self, form):
        messages.success(self.request, 'Account created successfully. Please login.')
        return super().form_valid(form)


class ProfileView(LoginRequiredMixin, DetailView):
    """
    View for user profile.
    """
    template_name = 'accounts/profile.html'
    model = User
    context_object_name = 'user_obj'
    
    def get_object(self):
        return self.request.user
    
    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context['profile'] = self.request.user.profile
        return context


class ProfileEditView(LoginRequiredMixin, UpdateView):
    """
    View for editing user profile.
    """
    template_name = 'accounts/profile_edit.html'
    model = UserProfile
    fields = ['organization', 'job_title', 'department', 'phone_number', 
              'avatar', 'theme_preference']
    success_url = reverse_lazy('accounts:profile')
    
    def get_object(self):
        return self.request.user.profile
    
    def form_valid(self, form):
        messages.success(self.request, 'Profile updated successfully.')
        return super().form_valid(form)