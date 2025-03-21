import streamlit as st
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Optional
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
import google.generativeai as genai
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from dotenv import load_dotenv

from app.components import document_view
from src.nlp.question_generator import QuestionGenerator
from src.utils.logger import get_logger

# Initialize logger
logger = get_logger(__name__)

# Load environment variables
load_dotenv()

# Configure Gemini API
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel('gemini-pro')
    GEMINI_AVAILABLE = True
except Exception as e:
    logger.error(f"Error configuring Gemini API: {str(e)}")
    GEMINI_AVAILABLE = False

# Model paths
BERT_MODEL_NAME = "bert-base-uncased"
FINE_TUNED_MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/question_generator")

# Initialize BERT models if available
try:
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
    if os.path.exists(FINE_TUNED_MODEL_PATH):
        question_classifier = AutoModelForSequenceClassification.from_pretrained(FINE_TUNED_MODEL_PATH)
        BERT_MODEL_AVAILABLE = True
    else:
        question_classifier = BertForSequenceClassification.from_pretrained(
            BERT_MODEL_NAME, num_labels=2)  # For relevance classification
        BERT_MODEL_AVAILABLE = True
except Exception as e:
    logger.error(f"Error loading BERT model: {str(e)}")
    BERT_MODEL_AVAILABLE = False

def render():
    """Render the interview questions page"""
    st.title("AI Interview Question Generator")
    st.write("Generate personalized interview questions based on resume and job description matches.")
    
    # Session state initialization
    if 'interview_questions' not in st.session_state:
        st.session_state.interview_questions = {}
    if 'candidate_responses' not in st.session_state:
        st.session_state.candidate_responses = {}
    if 'bert_training_data' not in st.session_state:
        st.session_state.bert_training_data = []
    
    # Check for required data
    if not st.session_state.get('resumes') or not st.session_state.get('job_descriptions'):
        st.warning("Please upload both resumes and job descriptions first.")
        return
    
    if not st.session_state.get('match_results'):
        st.warning("Please match resumes to job descriptions first.")
        return
    
    # Sidebar settings
    with st.sidebar:
        st.subheader("Question Generation Settings")
        num_questions = st.slider("Number of Questions", 3, 15, 8)
        question_types = st.multiselect(
            "Question Types",
            ["Technical", "Behavioral", "Experience", "Problem Solving", "Role-specific"],
            ["Technical", "Experience", "Problem Solving"]
        )
        difficulty = st.select_slider(
            "Question Difficulty",
            ["Basic", "Intermediate", "Advanced", "Expert"],
            "Intermediate"
        )
        
        # Model selection
        generation_options = ["Rule-based"]
        if GEMINI_AVAILABLE:
            generation_options.insert(0, "Gemini Pro")
        if BERT_MODEL_AVAILABLE:
            generation_options.insert(0, "BERT Fine-tuned")
            
        generation_model = st.radio(
            "Generation Model",
            options=generation_options,
            index=0 if BERT_MODEL_AVAILABLE else (0 if GEMINI_AVAILABLE else 0)
        )
        
        # Fine-tuning section
        if BERT_MODEL_AVAILABLE:
            with st.expander("BERT Model Fine-tuning"):
                if st.button("Fine-tune BERT Model"):
                    with st.spinner("Fine-tuning model..."):
                        if len(st.session_state.bert_training_data) >= 10:
                            success = fine_tune_bert_model(st.session_state.bert_training_data)
                            st.success("BERT model fine-tuned successfully!") if success else st.error("Fine-tuning failed")
                        else:
                            st.warning("Not enough training data (minimum 10 examples needed)")
    
    # Match selection
    st.header("Generate Interview Questions")
    st.subheader("Select a Resume-Job Match")
    
    if st.session_state.match_results:
        # Create dataframe for matches
        match_df = pd.DataFrame([
            {
                'Resume': result['resume_filename'],
                'Job': result['job_filename'],
                'Score': f"{result['score']:.2f}",
                'ID': f"{result['resume_id']}_{result['job_id']}"
            } for result in st.session_state.match_results
        ])
        match_df = match_df.sort_values('Score', ascending=False)
        
        # Display matches
        st.dataframe(match_df.drop(columns=['ID']), use_container_width=True)
        
        # Match selection
        selected_match = st.selectbox(
            "Select a match",
            options=match_df['ID'].tolist(),
            format_func=lambda x: f"{match_df[match_df['ID']==x]['Resume'].iloc[0]} - {match_df[match_df['ID']==x]['Job'].iloc[0]}"
        )
        
        # Extract resume and job IDs
        resume_id, job_id = selected_match.split('_', 1)
        
        # Find selected resume and job
        selected_resume = next((r for r in st.session_state.resumes if r['id'] == resume_id), None)
        selected_job = next((j for j in st.session_state.job_descriptions if j['id'] == job_id), None)
        
        if selected_resume and selected_job:
            col1, col2 = st.columns(2)
            with col1:
                document_view.render_document_card(selected_resume['filename'], 
                                                  selected_resume['processed']['clean_text'][:300] + "...", 
                                                  "Resume")
            with col2:
                document_view.render_document_card(selected_job['filename'], 
                                                  selected_job['processed']['clean_text'][:300] + "...", 
                                                  "Job Description")
            
            # Generate questions
            if st.button("Generate Interview Questions", key="gen_questions"):
                match_id = f"{resume_id}_{job_id}"
                
                with st.spinner("Generating personalized interview questions..."):
                    resume_text = selected_resume['processed']['clean_text']
                    job_text = selected_job['processed']['clean_text']
                    
                    # Generate questions based on selected model
                    if generation_model == "BERT Fine-tuned" and BERT_MODEL_AVAILABLE:
                        questions = generate_questions_bert(resume_text, job_text, num_questions, question_types, difficulty)
                    elif generation_model == "Gemini Pro" and GEMINI_AVAILABLE:
                        questions = generate_questions_gemini(resume_text, job_text, num_questions, question_types, difficulty)
                    else:
                        questions = generate_questions_rule_based(resume_text, job_text, num_questions, question_types, difficulty)
                    
                    # Store questions in session state
                    st.session_state.interview_questions[match_id] = {
                        'questions': questions,
                        'resume_filename': selected_resume['filename'],
                        'job_filename': selected_job['filename'],
                        'generation_time': time.time(),
                        'settings': {
                            'num_questions': num_questions,
                            'question_types': question_types,
                            'difficulty': difficulty,
                            'model': generation_model
                        }
                    }
                    
                    # Add to training data
                    for question in questions:
                        st.session_state.bert_training_data.append({
                            'resume_text': resume_text,
                            'job_text': job_text,
                            'question': question['question'],
                            'type': question['type'],
                            'difficulty': question['difficulty']
                        })
                
                st.success(f"Generated {len(questions)} interview questions")
            
            # Display questions if available
            match_id = f"{resume_id}_{job_id}"
            if match_id in st.session_state.interview_questions:
                display_interview_questions(match_id)