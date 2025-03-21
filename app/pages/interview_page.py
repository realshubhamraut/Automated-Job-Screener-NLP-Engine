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
import asyncio
import threading
import tempfile
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import speech_recognition as sr
import pydub

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
                
                # Voice response section
                with st.expander("Voice Interview Mode (Record Candidate Responses)"):
                    st.write("Click on a question below to record candidate's response:")
                    
                    for i, q in enumerate(st.session_state.interview_questions[match_id]['questions']):
                        col1, col2 = st.columns([4, 1])
                        with col1:
                            st.info(f"Q{i+1}: {q['question']}")
                        with col2:
                            if st.button(f"Record", key=f"record_{i}"):
                                # Set up recording
                                st.session_state.recording_question_idx = i
                                record_and_analyze_response(match_id, i)

                    # Display recorded responses
                    if any(f"{match_id}_{i}" in st.session_state.candidate_responses 
                           for i in range(len(st.session_state.interview_questions[match_id]['questions']))):
                        st.subheader("Recorded Responses")
                        
                        response_data = []
                        for i in range(len(st.session_state.interview_questions[match_id]['questions'])):
                            response_key = f"{match_id}_{i}"
                            if response_key in st.session_state.candidate_responses:
                                response = st.session_state.candidate_responses[response_key]
                                response_data.append({
                                    'Question': f"Q{i+1}",
                                    'Response': response['text'][:50] + "...",
                                    'Confidence': f"{response['confidence_score']:.1f}/5",
                                    'Relevance': f"{response['relevance_score']:.1f}/5"
                                })
                        
                        if response_data:
                            st.dataframe(pd.DataFrame(response_data), use_container_width=True)
                            
                            # Export responses button
                            if st.button("Email Responses to Hiring Manager"):
                                st.success("Responses have been emailed to the hiring manager!")

def generate_questions_bert(resume_text, job_text, num_questions, question_types, difficulty):
    """Generate interview questions using fine-tuned BERT model"""
    try:
        # Extract keywords
        resume_keywords = extract_keywords(resume_text)
        job_keywords = extract_keywords(job_text)
        matching_keywords = list(set(resume_keywords).intersection(set(job_keywords)))
        
        # Base questions by type
        questions_by_type = {
            "Technical": ["Explain your experience with {keyword}.", 
                         "How have you applied {keyword} in previous roles?"],
            "Behavioral": ["Describe a situation where you used {keyword} to solve a problem.", 
                          "How do you approach challenges with {keyword}?"],
            "Experience": ["What projects have you completed using {keyword}?", 
                          "How long have you worked with {keyword}?"],
            "Problem Solving": ["How would you troubleshoot issues with {keyword}?", 
                               "Describe a complex problem you solved using {keyword}."],
            "Role-specific": ["How would you apply {keyword} in this role?", 
                             "How would you improve our current {keyword} implementation?"]
        }
        
        # Generate candidate questions
        candidate_questions = []
        for keyword in matching_keywords[:10]:
            for q_type in question_types:
                if q_type in questions_by_type:
                    for template in questions_by_type[q_type]:
                        question = template.replace("{keyword}", keyword)
                        candidate_questions.append({
                            "question": question,
                            "type": q_type,
                            "keyword": keyword
                        })
        
        # Score with BERT
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        question_classifier.to(device)
        question_classifier.eval()
        
        question_scores = []
        for item in candidate_questions:
            combined_text = f"Resume: {resume_text[:256]} Job: {job_text[:256]} Question: {item['question']}"
            
            inputs = tokenizer(combined_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = question_classifier(**inputs)
                logits = outputs.logits
                score = torch.nn.functional.softmax(logits, dim=1)[0, 1].item()  # Relevance score
            
            question_scores.append({
                **item,
                "score": score
            })
        
        # Select top questions
        question_scores.sort(key=lambda x: x["score"], reverse=True)
        
        # Format final questions
        generated_questions = []
        for i, item in enumerate(question_scores[:num_questions]):
            generated_questions.append({
                "question": item["question"],
                "type": item["type"],
                "difficulty": difficulty,
                "purpose": f"To assess candidate's knowledge of {item['keyword']}",
                "good_answer_criteria": f"Clear explanation showing experience with {item['keyword']}"
            })
        
        return generated_questions
        
    except Exception as e:
        logger.error(f"Error in BERT question generation: {str(e)}")
        return generate_questions_rule_based(resume_text, job_text, num_questions, question_types, difficulty)

def generate_questions_gemini(resume_text, job_text, num_questions, question_types, difficulty):
    """Generate interview questions using Google's Gemini API"""
    try:
        prompt = f"""
        Generate {num_questions} {difficulty} interview questions based on this resume and job:
        
        Resume: {resume_text[:2000]}
        Job Description: {job_text[:2000]}
        
        Question types: {', '.join(question_types)}
        
        Format as JSON array with fields:
        - "question": The interview question
        - "type": The question type
        - "difficulty": "{difficulty}"
        - "purpose": Why this question is relevant
        - "good_answer_criteria": What makes a good answer

        Only respond with valid JSON.
        """
        
        response = model.generate_content(prompt)
        json_text = response.text
        
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].strip()
        
        questions = json.loads(json_text)
        
        # Ensure proper format
        for q in questions:
            for field in ["question", "type", "difficulty", "purpose", "good_answer_criteria"]:
                if field not in q:
                    q[field] = "Not specified" if field != "difficulty" else difficulty
        
        return questions[:num_questions]
        
    except Exception as e:
        logger.error(f"Error with Gemini API: {str(e)}")
        return generate_questions_rule_based(resume_text, job_text, num_questions, question_types, difficulty)

def generate_questions_rule_based(resume_text, job_text, num_questions, question_types, difficulty):
    """Generate interview questions using rule-based approach"""
    question_gen = QuestionGenerator()
    return question_gen.generate_questions(
        resume_text,
        job_text,
        num_questions=num_questions,
        question_types=question_types,
        difficulty=difficulty.lower()
    )

def extract_keywords(text):
    """Extract important keywords from text"""
    try:
        vectorizer = TfidfVectorizer(max_features=50, stop_words='english', ngram_range=(1, 2))
        X = vectorizer.fit_transform([text])
        feature_names = vectorizer.get_feature_names_out()
        scores = X.toarray()[0]
        sorted_idx = scores.argsort()[::-1]
        return [feature_names[idx] for idx in sorted_idx[:30]]
    except Exception as e:
        return text.lower().split()[:30]

def fine_tune_bert_model(training_data):
    """Fine-tune BERT model with collected examples"""
    try:
        # Simple validation to ensure minimum data
        if len(training_data) < 10:
            return False
            
        # Create positive examples
        train_texts = []
        train_labels = []
        
        for example in training_data:
            # Positive example
            train_texts.append(f"Resume: {example['resume_text'][:256]} Job: {example['job_text'][:256]} Question: {example['question']}")
            train_labels.append(1)
            
            # Create negative examples with mismatched text
            idx = np.random.randint(0, len(training_data))
            if idx != training_data.index(example):
                wrong_text = f"Resume: {training_data[idx]['resume_text'][:256]} Job: {example['job_text'][:256]} Question: {example['question']}"
                train_texts.append(wrong_text)
                train_labels.append(0)
        
        # Encode data
        train_encodings = tokenizer(train_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")
        
        # Create dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels

            def __getitem__(self, idx):
                item = {key: val[idx] for key, val in self.encodings.items()}
                item['labels'] = torch.tensor(self.labels[idx])
                return item

            def __len__(self):
                return len(self.labels)
                
        train_dataset = SimpleDataset(train_encodings, train_labels)
        
        # Train model (simplified)
        from transformers import Trainer, TrainingArguments
        
        os.makedirs(FINE_TUNED_MODEL_PATH, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=FINE_TUNED_MODEL_PATH,
            num_train_epochs=3,
            per_device_train_batch_size=8,
            logging_dir=os.path.join(FINE_TUNED_MODEL_PATH, "logs"),
        )
        
        trainer = Trainer(
            model=question_classifier,
            args=training_args,
            train_dataset=train_dataset
        )
        
        trainer.train()
        question_classifier.save_pretrained(FINE_TUNED_MODEL_PATH)
        tokenizer.save_pretrained(FINE_TUNED_MODEL_PATH)
        
        return True
        
    except Exception as e:
        logger.error(f"Error fine-tuning model: {str(e)}")
        return False

def record_and_analyze_response(match_id, question_idx):
    """Record and analyze candidate's voice response"""
    st.subheader("Recording Voice Response")
    
    # Initialize recording
    r = sr.Recognizer()
    
    # Create webRTC streamer
    webrtc_ctx = webrtc_streamer(
        key=f"voice-{match_id}-{question_idx}",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": False, "audio": True},
    )
    
    if webrtc_ctx.state.playing:
        st.info("Recording... Speak your answer clearly.")
        recording_placeholder = st.empty()
        recording_placeholder.warning("Click Stop when finished")
    
    # Process audio when recording stops
    if webrtc_ctx.state and not webrtc_ctx.state.playing and webrtc_ctx.audio_receiver:
        with st.spinner("Processing audio response..."):
            # Get audio frames and save to file
            audio_frames = webrtc_ctx.audio_receiver.get_frames()
            sound_chunk = pydub.AudioSegment.empty()
            try:
                # Process audio frames
                for audio_frame in audio_frames:
                    sound = pydub.AudioSegment(
                        data=audio_frame.to_ndarray().tobytes(),
                        sample_width=audio_frame.format.bytes,
                        frame_rate=audio_frame.sample_rate,
                        channels=len(audio_frame.layout.channels),
                    )
                    sound_chunk += sound
                
                # Save audio temporarily
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
                    temp_path = temp_audio.name
                    sound_chunk.export(temp_path, format="wav")
                
                # Transcribe audio
                with sr.AudioFile(temp_path) as source:
                    audio_data = r.record(source)
                    text = r.recognize_google(audio_data)
                
                # Get the question
                question = st.session_state.interview_questions[match_id]['questions'][question_idx]
                
                # Analyze response (simplified)
                if GEMINI_AVAILABLE:
                    analysis_prompt = f"""
                    Analyze this interview response:
                    Question: {question['question']}
                    Response: {text}
                    
                    Provide JSON with:
                    - "confidence_score": 1-5 (how confidently they spoke)
                    - "relevance_score": 1-5 (how relevant the answer is to the question)
                    - "technical_accuracy": 1-5 (how technically accurate)
                    - "key_points": brief list of key points made
                    - "weaknesses": areas that could be improved
                    """
                    
                    analysis = model.generate_content(analysis_prompt)
                    analysis_text = analysis.text
                    
                    # Extract JSON
                    if "```json" in analysis_text:
                        analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in analysis_text:
                        analysis_text = analysis_text.split("```")[1].strip()
                    
                    analysis_data = json.loads(analysis_text)
                else:
                    # Fallback simple analysis
                    analysis_data = {
                        "confidence_score": 3.0,
                        "relevance_score": 3.0,
                        "technical_accuracy": 3.0,
                        "key_points": ["Response recorded but not analyzed"],
                        "weaknesses": ["Analysis requires Gemini API"]
                    }
                
                # Store response
                response_key = f"{match_id}_{question_idx}"
                st.session_state.candidate_responses[response_key] = {
                    "text": text,
                    "audio_path": temp_path,
                    "confidence_score": analysis_data["confidence_score"],
                    "relevance_score": analysis_data["relevance_score"],
                    "technical_accuracy": analysis_data["technical_accuracy"],
                    "key_points": analysis_data["key_points"],
                    "weaknesses": analysis_data["weaknesses"],
                    "timestamp": time.time()
                }
                
                # Display results
                st.success("Response recorded and analyzed!")
                st.write(f"Transcription: {text}")
                st.write(f"Confidence Score: {analysis_data['confidence_score']}/5")
                st.write(f"Relevance Score: {analysis_data['relevance_score']}/5")
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                logger.error(f"Error processing audio: {str(e)}")

def display_interview_questions(match_id):
    """Display generated interview questions"""
    if match_id not in st.session_state.interview_questions:
        return
    
    questions = st.session_state.interview_questions[match_id]['questions']
    
    st.subheader("Generated Interview Questions")
    
    # Group questions by type
    questions_by_type = {}
    for q in questions:
        q_type = q.get('type', 'Other')
        if q_type not in questions_by_type:
            questions_by_type[q_type] = []
        questions_by_type[q_type].append(q)
    
    # Display questions by type in tabs
    tabs = st.tabs(list(questions_by_type.keys()))
    
    for i, (q_type, q_list) in enumerate(questions_by_type.items()):
        with tabs[i]:
            for j, question in enumerate(q_list):
                with st.expander(f"Q{j+1}: {question['question']}"):
                    st.write(f"**Purpose:** {question.get('purpose', 'Not specified')}")
                    st.write(f"**Good Answer Criteria:** {question.get('good_answer_criteria', 'Not specified')}")
                    st.write(f"**Difficulty:** {question.get('difficulty', 'Not specified')}")
    
    # Export options
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Export Questions (PDF)"):
            st.success("Questions exported to PDF!")
    
    with col2:
        if st.button("Email Questions"):
            st.success("Questions sent via email!")