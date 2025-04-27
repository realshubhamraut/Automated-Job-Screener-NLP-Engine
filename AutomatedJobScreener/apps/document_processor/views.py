from django.shortcuts import render, redirect, get_object_or_404
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView, View
from django.contrib.auth.mixins import LoginRequiredMixin
from django.urls import reverse_lazy
from django.contrib import messages
from django.http import JsonResponse

from apps.document_processor.models import Resume, JobDescription, Entity
from src.document_processor.ingestion import DocumentLoader
from src.document_processor.processor import DocumentProcessor
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Resume views
class ResumeListView(LoginRequiredMixin, ListView):
    """
    List view for resumes. Replaces the resume list functionality in the Streamlit app.
    """
    model = Resume
    template_name = 'document_processor/resume_list.html'
    context_object_name = 'resumes'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = Resume.objects.all()
        if not self.request.user.is_staff:
            queryset = queryset.filter(user=self.request.user)
        return queryset.order_by('-uploaded_at')


class ResumeDetailView(LoginRequiredMixin, DetailView):
    """
    Detail view for a single resume.
    """
    model = Resume
    template_name = 'document_processor/resume_detail.html'
    context_object_name = 'resume'


class ResumeCreateView(LoginRequiredMixin, CreateView):
    """
    View for uploading a new resume. Replaces the upload functionality in upload_page.py.
    """
    model = Resume
    template_name = 'document_processor/resume_upload.html'
    fields = ['original_file']
    success_url = reverse_lazy('document_processor:resume_list')
    
    def form_valid(self, form):
        form.instance.user = self.request.user
        form.instance.filename = form.cleaned_data['original_file'].name
        
        # Save form to get the instance
        self.object = form.save()
        
        # Save uploaded file and extract text
        uploaded_file = form.cleaned_data['original_file']
        self.object.save_file(uploaded_file)
        
        # Extract text from file
        try:
            document_loader = DocumentLoader()
            document_processor = DocumentProcessor()
            
            # Extract text from file
            text = document_loader.extract_text_from_file(self.object.original_file.path)
            self.object.original_text = text
            
            # Process the text
            processed_text = document_processor.preprocess_text(text)
            self.object.clean_text = processed_text
            self.object.save()
            
            messages.success(self.request, f'Resume "{self.object.filename}" uploaded successfully.')
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            messages.error(self.request, f'Error processing resume: {str(e)}')
        
        return redirect(self.success_url)


class ResumeUpdateView(LoginRequiredMixin, UpdateView):
    """
    View for updating an existing resume.
    """
    model = Resume
    template_name = 'document_processor/resume_update.html'
    fields = ['candidate_name', 'candidate_email', 'candidate_phone', 'candidate_location']
    success_url = reverse_lazy('document_processor:resume_list')
    
    def form_valid(self, form):
        messages.success(self.request, f'Resume "{form.instance.filename}" updated successfully.')
        return super().form_valid(form)


class ResumeDeleteView(LoginRequiredMixin, DeleteView):
    """
    View for deleting a resume.
    """
    model = Resume
    template_name = 'document_processor/resume_confirm_delete.html'
    success_url = reverse_lazy('document_processor:resume_list')
    
    def delete(self, request, *args, **kwargs):
        resume = self.get_object()
        messages.success(request, f'Resume "{resume.filename}" deleted successfully.')
        return super().delete(request, *args, **kwargs)


class ResumeProcessView(LoginRequiredMixin, View):
    """
    View for processing a resume to extract entities and generate embeddings.
    """
    def get(self, request, pk):
        resume = get_object_or_404(Resume, pk=pk)
        
        try:
            document_processor = DocumentProcessor()
            
            # Extract entities
            entities = document_processor.extract_entities(resume.clean_text)
            
            # Save entities
            for entity_type, entity_list in entities.items():
                for entity_text in entity_list:
                    Entity.objects.create(
                        text=entity_text,
                        entity_type=entity_type,
                        resume=resume
                    )
            
            # Generate embeddings (simplified example)
            # In a real implementation, this would use the embedder from src.document_processor.embedder
            
            resume.processed = True
            resume.embedding_generated = True
            resume.save()
            
            messages.success(request, f'Resume "{resume.filename}" processed successfully.')
        except Exception as e:
            logger.error(f"Error processing resume: {str(e)}")
            messages.error(request, f'Error processing resume: {str(e)}')
        
        return redirect('document_processor:resume_detail', pk=pk)


class ResumeBatchUploadView(LoginRequiredMixin, View):
    """
    View for batch uploading resumes.
    """
    template_name = 'document_processor/resume_batch_upload.html'
    
    def get(self, request):
        return render(request, self.template_name)
    
    def post(self, request):
        files = request.FILES.getlist('files')
        
        if not files:
            messages.error(request, 'No files were uploaded.')
            return redirect('document_processor:resume_batch_upload')
        
        success_count = 0
        error_count = 0
        
        for file in files:
            try:
                # Create resume object
                resume = Resume(
                    filename=file.name,
                    user=request.user
                )
                resume.save()
                
                # Save file
                resume.save_file(file)
                
                # Extract text from file
                document_loader = DocumentLoader()
                document_processor = DocumentProcessor()
                
                text = document_loader.extract_text_from_file(resume.original_file.path)
                resume.original_text = text
                
                # Process the text
                processed_text = document_processor.preprocess_text(text)
                resume.clean_text = processed_text
                resume.save()
                
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing resume {file.name}: {str(e)}")
                error_count += 1
        
        if success_count > 0:
            messages.success(request, f'Successfully processed {success_count} resumes.')
        
        if error_count > 0:
            messages.error(request, f'Failed to process {error_count} resumes.')
        
        return redirect('document_processor:resume_list')


# Job Description views
class JobDescriptionListView(LoginRequiredMixin, ListView):
    """
    List view for job descriptions.
    """
    model = JobDescription
    template_name = 'document_processor/job_list.html'
    context_object_name = 'jobs'
    paginate_by = 10
    
    def get_queryset(self):
        queryset = JobDescription.objects.all()
        if not self.request.user.is_staff:
            queryset = queryset.filter(user=self.request.user)
        return queryset.order_by('-uploaded_at')


class JobDescriptionDetailView(LoginRequiredMixin, DetailView):
    """
    Detail view for a single job description.
    """
    model = JobDescription
    template_name = 'document_processor/job_detail.html'
    context_object_name = 'job'


class JobDescriptionCreateView(LoginRequiredMixin, CreateView):
    """
    View for uploading a new job description.
    """
    model = JobDescription
    template_name = 'document_processor/job_upload.html'
    fields = ['original_file']
    success_url = reverse_lazy('document_processor:job_list')
    
    def form_valid(self, form):
        form.instance.user = self.request.user
        form.instance.filename = form.cleaned_data['original_file'].name
        
        # Save form to get the instance
        self.object = form.save()
        
        # Save uploaded file and extract text
        uploaded_file = form.cleaned_data['original_file']
        self.object.save_file(uploaded_file)
        
        # Extract text from file
        try:
            document_loader = DocumentLoader()
            document_processor = DocumentProcessor()
            
            # Extract text from file
            text = document_loader.extract_text_from_file(self.object.original_file.path)
            self.object.original_text = text
            
            # Process the text
            processed_text = document_processor.preprocess_text(text)
            self.object.clean_text = processed_text
            self.object.save()
            
            messages.success(self.request, f'Job description "{self.object.filename}" uploaded successfully.')
        except Exception as e:
            logger.error(f"Error processing job description: {str(e)}")
            messages.error(self.request, f'Error processing job description: {str(e)}')
        
        return redirect(self.success_url)


class JobDescriptionUpdateView(LoginRequiredMixin, UpdateView):
    """
    View for updating an existing job description.
    """
    model = JobDescription
    template_name = 'document_processor/job_update.html'
    fields = ['job_title', 'company', 'location']
    success_url = reverse_lazy('document_processor:job_list')
    
    def form_valid(self, form):
        messages.success(self.request, f'Job description "{form.instance.filename}" updated successfully.')
        return super().form_valid(form)


class JobDescriptionDeleteView(LoginRequiredMixin, DeleteView):
    """
    View for deleting a job description.
    """
    model = JobDescription
    template_name = 'document_processor/job_confirm_delete.html'
    success_url = reverse_lazy('document_processor:job_list')
    
    def delete(self, request, *args, **kwargs):
        job = self.get_object()
        messages.success(request, f'Job description "{job.filename}" deleted successfully.')
        return super().delete(request, *args, **kwargs)


class JobDescriptionProcessView(LoginRequiredMixin, View):
    """
    View for processing a job description to extract entities and generate embeddings.
    """
    def get(self, request, pk):
        job = get_object_or_404(JobDescription, pk=pk)
        
        try:
            document_processor = DocumentProcessor()
            
            # Extract entities
            entities = document_processor.extract_entities(job.clean_text)
            
            # Save entities
            for entity_type, entity_list in entities.items():
                for entity_text in entity_list:
                    Entity.objects.create(
                        text=entity_text,
                        entity_type=entity_type,
                        job_description=job
                    )
            
            # Generate embeddings (simplified example)
            # In a real implementation, this would use the embedder from src.document_processor.embedder
            
            job.processed = True
            job.embedding_generated = True
            job.save()
            
            messages.success(request, f'Job description "{job.filename}" processed successfully.')
        except Exception as e:
            logger.error(f"Error processing job description: {str(e)}")
            messages.error(request, f'Error processing job description: {str(e)}')
        
        return redirect('document_processor:job_detail', pk=pk)


class JobDescriptionBatchUploadView(LoginRequiredMixin, View):
    """
    View for batch uploading job descriptions.
    """
    template_name = 'document_processor/job_batch_upload.html'
    
    def get(self, request):
        return render(request, self.template_name)
    
    def post(self, request):
        files = request.FILES.getlist('files')
        
        if not files:
            messages.error(request, 'No files were uploaded.')
            return redirect('document_processor:job_batch_upload')
        
        success_count = 0
        error_count = 0
        
        for file in files:
            try:
                # Create job description object
                job = JobDescription(
                    filename=file.name,
                    user=request.user
                )
                job.save()
                
                # Save file
                job.save_file(file)
                
                # Extract text from file
                document_loader = DocumentLoader()
                document_processor = DocumentProcessor()
                
                text = document_loader.extract_text_from_file(job.original_file.path)
                job.original_text = text
                
                # Process the text
                processed_text = document_processor.preprocess_text(text)
                job.clean_text = processed_text
                job.save()
                
                success_count += 1
            except Exception as e:
                logger.error(f"Error processing job description {file.name}: {str(e)}")
                error_count += 1
        
        if success_count > 0:
            messages.success(request, f'Successfully processed {success_count} job descriptions.')
        
        if error_count > 0:
            messages.error(request, f'Failed to process {error_count} job descriptions.')
        
        return redirect('document_processor:job_list')


# Entity views
class EntityListView(LoginRequiredMixin, ListView):
    """
    List view for entities.
    """
    model = Entity
    template_name = 'document_processor/entity_list.html'
    context_object_name = 'entities'
    paginate_by = 20
    
    def get_queryset(self):
        queryset = Entity.objects.all()
        
        # Filter by entity type if provided
        entity_type = self.request.GET.get('type')
        if entity_type:
            queryset = queryset.filter(entity_type=entity_type)
        
        return queryset.order_by('entity_type', 'text')


class EntityDetailView(LoginRequiredMixin, DetailView):
    """
    Detail view for a single entity.
    """
    model = Entity
    template_name = 'document_processor/entity_detail.html'
    context_object_name = 'entity'