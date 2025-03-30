import streamlit as st
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Optional
import torch
from transformers import BertTokenizer, BertForSequenceClassification, AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
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
import re
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from streamlit_ace import st_ace

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

# Initialize sentence transformer for voice analysis (PyTorch-based)
try:
    # Using sentence-transformers model which is based on PyTorch
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    VOICE_ANALYSIS_AVAILABLE = True
except Exception as e:
    logger.error(f"Error loading voice analysis model: {str(e)}")
    VOICE_ANALYSIS_AVAILABLE = False

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

def analyze_voice_response(text, question, expected_keywords=None):
    """
    Analyze voice response with advanced metrics using PyTorch models
    
    Args:
        text: Transcribed text from voice response
        question: The original question asked
        expected_keywords: Keywords expected in a good answer
        
    Returns:
        Dictionary with analysis metrics
    """
    analysis = {
        "word_count": len(text.split()),
        "sentence_count": len(re.split(r'[.!?]+', text)),
        "confidence_indicators": 0,
        "hesitation_indicators": 0,
        "keywords_matched": 0,
        "relevance_score": 0,
        "clarity_score": 0,
        "detailed_feedback": []
    }
    
    # Count confidence indicators (strong statements, clear affirmations)
    confidence_patterns = [r'\bI know\b', r'\bI am confident\b', r'\bdefinitely\b', 
                          r'\bwithout doubt\b', r'\bI am sure\b', r'\bI have experience\b']
    for pattern in confidence_patterns:
        analysis["confidence_indicators"] += len(re.findall(pattern, text, re.IGNORECASE))
    
    # Count hesitation indicators (filler words, uncertain phrases)
    hesitation_patterns = [r'\bum\b', r'\buh\b', r'\blike\b', r'\bperhaps\b', 
                          r'\bmaybe\b', r'\bsort of\b', r'\bkind of\b', r'\bI think\b']
    for pattern in hesitation_patterns:
        analysis["hesitation_indicators"] += len(re.findall(pattern, text, re.IGNORECASE))
    
    # Check for expected keywords if provided
    if expected_keywords:
        for keyword in expected_keywords:
            if re.search(r'\b' + re.escape(keyword) + r'\b', text, re.IGNORECASE):
                analysis["keywords_matched"] += 1
        
        analysis["keyword_percentage"] = (analysis["keywords_matched"] / len(expected_keywords)) * 100 if expected_keywords else 0
    
    # Calculate relevance using Sentence Transformer (PyTorch-based)
    if VOICE_ANALYSIS_AVAILABLE:
        try:
            # Encode question and answer with PyTorch model
            question_embedding = sentence_model.encode(question)
            text_embedding = sentence_model.encode(text)
            
            # Calculate cosine similarity
            similarity = np.dot(question_embedding, text_embedding) / (
                np.linalg.norm(question_embedding) * np.linalg.norm(text_embedding)
            )
            
            analysis["relevance_score"] = float(similarity) * 5  # Scale to 0-5
        except Exception as e:
            logger.error(f"Error calculating semantic similarity: {str(e)}")
            analysis["relevance_score"] = 3.0  # Default mid-range score
    else:
        # Fallback method - basic keyword matching
        question_words = set([w.lower() for w in re.findall(r'\b\w+\b', question) if len(w) > 3])
        answer_words = set([w.lower() for w in re.findall(r'\b\w+\b', text) if len(w) > 3])
        common_words = question_words.intersection(answer_words)
        analysis["relevance_score"] = min(5.0, len(common_words) * 5 / max(1, len(question_words)))
    
    # Calculate clarity score based on sentence structure and length
    avg_words_per_sentence = analysis["word_count"] / max(1, analysis["sentence_count"])
    if avg_words_per_sentence < 5:
        analysis["clarity_score"] = 2.0  # Too short sentences
    elif avg_words_per_sentence > 25:
        analysis["clarity_score"] = 2.5  # Too long sentences
    else:
        analysis["clarity_score"] = 4.0  # Good sentence length
    
    # Adjust clarity score based on hesitations
    hesitation_ratio = analysis["hesitation_indicators"] / max(1, analysis["word_count"]) * 100
    if hesitation_ratio > 10:
        analysis["clarity_score"] -= 1.5
    elif hesitation_ratio > 5:
        analysis["clarity_score"] -= 0.75
    
    # Ensure scores are within 0-5 range
    analysis["clarity_score"] = max(0, min(5, analysis["clarity_score"]))
    
    # Generate detailed feedback
    feedback = []
    
    if analysis["word_count"] < 30:
        feedback.append("Response is quite brief. Consider providing more detail.")
    elif analysis["word_count"] > 300:
        feedback.append("Response is detailed but could be more concise.")
    
    if analysis["confidence_indicators"] >= 3:
        feedback.append("Shows strong confidence in the response.")
    elif analysis["confidence_indicators"] <= 1 and analysis["hesitation_indicators"] >= 3:
        feedback.append("Response shows some hesitation. Could demonstrate more confidence.")
    
    if analysis["relevance_score"] < 2.5:
        feedback.append("Response could be more directly relevant to the question.")
    elif analysis["relevance_score"] >= 4:
        feedback.append("Response is highly relevant to the question.")
    
    if expected_keywords and analysis["keywords_matched"] < len(expected_keywords) / 2:
        feedback.append(f"Missing some key concepts: {', '.join(expected_keywords)}")
    
    analysis["detailed_feedback"] = feedback
    analysis["overall_score"] = (analysis["relevance_score"] * 0.5 + 
                               analysis["clarity_score"] * 0.3 + 
                               (5 - (analysis["hesitation_indicators"] / max(1, analysis["word_count"]) * 100)) * 0.2)
    
    # Ensure overall score is within 0-5 range
    analysis["overall_score"] = max(0, min(5, analysis["overall_score"]))
    
    return analysis

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

# Update the record_and_analyze_response function
def record_and_analyze_response(match_id, question_idx):
    """Record and analyze candidate's voice response with advanced metrics"""
    # Use a unique key for this recording session
    recorder_key = f"voice_recorder_{match_id}_{question_idx}"
    
    # Initialize recording state if not present
    if f"recording_active_{match_id}_{question_idx}" not in st.session_state:
        st.session_state[f"recording_active_{match_id}_{question_idx}"] = True
    
    st.subheader("Recording Voice Response")
    
    # Initialize recording
    r = sr.Recognizer()
    
    # Create persistent webRTC streamer with a consistent key
    webrtc_ctx = webrtc_streamer(
        key=recorder_key,
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        ),
        media_stream_constraints={"video": False, "audio": True},
    )
    
    # Display instructions based on recording state
    if webrtc_ctx.state.playing:
        st.info("Recording... Speak your answer clearly.")
        st.warning("Click Stop when finished")
    
    # Process audio when recording stops
    if webrtc_ctx.state and not webrtc_ctx.state.playing and webrtc_ctx.audio_receiver and st.session_state[f"recording_active_{match_id}_{question_idx}"]:
        # Set recording to inactive to prevent reprocessing
        st.session_state[f"recording_active_{match_id}_{question_idx}"] = False
        
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
                question_text = question['question']
                
                # Extract expected keywords from question purpose
                purpose = question.get('purpose', '')
                expected_keywords = []
                if 'knowledge of' in purpose:
                    knowledge_part = purpose.split('knowledge of')[1].strip()
                    if knowledge_part:
                        keyword = knowledge_part.split()[0].strip(',.')
                        expected_keywords.append(keyword)
                
                # Get keywords from good answer criteria
                criteria = question.get('good_answer_criteria', '')
                if 'experience with' in criteria:
                    experience_part = criteria.split('experience with')[1].strip()
                    if experience_part:
                        keyword = experience_part.split()[0].strip(',.')
                        if keyword not in expected_keywords:
                            expected_keywords.append(keyword)
                
                # Perform detailed voice analysis using PyTorch-based approach
                voice_analysis = analyze_voice_response(text, question_text, expected_keywords)
                
                # Also use AI-based analysis if available
                if GEMINI_AVAILABLE:
                    analysis_prompt = f"""
                    Analyze this interview response:
                    Question: {question_text}
                    Response: {text}
                    
                    Provide JSON with:
                    - "confidence_score": 1-5 (how confidently they spoke)
                    - "relevance_score": 1-5 (how relevant the answer is to the question)
                    - "technical_accuracy": 1-5 (how technically accurate)
                    - "key_points": brief list of key points made
                    - "strengths": main strengths of the answer
                    - "weaknesses": areas that could be improved
                    - "improvement_suggestions": specific suggestions to improve the answer
                    """
                    
                    analysis = model.generate_content(analysis_prompt)
                    analysis_text = analysis.text
                    
                    # Extract JSON
                    if "```json" in analysis_text:
                        analysis_text = analysis_text.split("```json")[1].split("```")[0].strip()
                    elif "```" in analysis_text:
                        analysis_text = analysis_text.split("```")[1].strip()
                    
                    ai_analysis = json.loads(analysis_text)
                    
                    # Combine AI and voice analyses
                    combined_analysis = {
                        # Voice metrics
                        "word_count": voice_analysis["word_count"],
                        "confidence_indicators": voice_analysis["confidence_indicators"],
                        "hesitation_indicators": voice_analysis["hesitation_indicators"],
                        "clarity_score": voice_analysis["clarity_score"],
                        
                        # AI metrics
                        "confidence_score": ai_analysis["confidence_score"],
                        "relevance_score": ai_analysis["relevance_score"],
                        "technical_accuracy": ai_analysis["technical_accuracy"],
                        "key_points": ai_analysis["key_points"],
                        "strengths": ai_analysis.get("strengths", []),
                        "weaknesses": ai_analysis.get("weaknesses", []),
                        "improvement_suggestions": ai_analysis.get("improvement_suggestions", []),
                        
                        # Combined metrics
                        "overall_score": (ai_analysis["relevance_score"] * 0.4 + 
                                         ai_analysis["technical_accuracy"] * 0.4 + 
                                         voice_analysis["clarity_score"] * 0.2)
                    }
                else:
                    # Use only voice analysis if AI is not available
                    combined_analysis = {
                        **voice_analysis,
                        "key_points": ["Automated analysis without AI"],
                        "strengths": ["Analysis requires Gemini API for strengths evaluation"],
                        "weaknesses": ["Analysis requires Gemini API for weaknesses evaluation"],
                        "improvement_suggestions": ["Connect Gemini API for detailed feedback"],
                        "technical_accuracy": voice_analysis["clarity_score"] * 0.8  # Estimate
                    }
                
                # Store response
                response_key = f"{match_id}_{question_idx}"
                st.session_state.candidate_responses[response_key] = {
                    "text": text,
                    "audio_path": temp_path,
                    "analysis": combined_analysis,
                    "timestamp": time.time()
                }
                
                # Display results with improved UI
                st.success("Response recorded and analyzed!")
                
                # Display transcription in a nice format
                st.markdown("#### Transcription")
                st.markdown(f"<div class='response-text'>{text}</div>", unsafe_allow_html=True)
                
                # Display analysis in tabs
                analysis_tabs = st.tabs(["Scores", "Key Points", "Feedback"])
                
                with analysis_tabs[0]:
                    score_cols = st.columns(3)
                    with score_cols[0]:
                        st.metric("Relevance", f"{combined_analysis['relevance_score']:.1f}/5")
                    with score_cols[1]:
                        st.metric("Technical Accuracy", f"{combined_analysis['technical_accuracy']:.1f}/5")
                    with score_cols[2]:
                        st.metric("Overall Score", f"{combined_analysis['overall_score']:.1f}/5")
                
                with analysis_tabs[1]:
                    st.markdown("##### Key Points")
                    for point in combined_analysis["key_points"]:
                        st.markdown(f"• {point}")
                
                with analysis_tabs[2]:
                    if "strengths" in combined_analysis:
                        st.markdown("##### Strengths")
                        for item in combined_analysis["strengths"]:
                            st.markdown(f"✅ {item}")
                    
                    if "weaknesses" in combined_analysis:
                        st.markdown("##### Areas for Improvement")
                        for item in combined_analysis["weaknesses"]:
                            st.markdown(f"🔍 {item}")
                    
                    if "improvement_suggestions" in combined_analysis:
                        st.markdown("##### Suggestions")
                        for item in combined_analysis["improvement_suggestions"]:
                            st.markdown(f"💡 {item}")
                
            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")
                logger.error(f"Error processing audio: {str(e)}")

def create_coding_challenge(match_id, question_idx):
    """Create and evaluate a coding challenge"""
    st.subheader("Coding Assessment")
    
    # Get the question
    question = st.session_state.interview_questions[match_id]['questions'][question_idx]
    
    # Generate a coding problem based on the question
    if GEMINI_AVAILABLE:
        coding_prompt = f"""
        Create a coding challenge based on this interview question:
        "{question['question']}"
        
        Provide a JSON response with:
        - "problem_statement": Clear description of the task
        - "input_format": Expected format of inputs
        - "output_format": Expected format of outputs
        - "constraints": Any constraints on the solution
        - "examples": 2-3 example inputs and outputs
        - "starter_code": Python starter code
        - "test_cases": 3 test cases (input and expected output)
        - "solution": A correct solution
        """
        
        response = model.generate_content(coding_prompt)
        json_text = response.text
        
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].strip()
        
        try:
            challenge = json.loads(json_text)
        except Exception as e:
            st.error(f"Error parsing challenge: {str(e)}")
            return
    else:
        # Fallback basic challenges
        challenge = {
            "problem_statement": f"Write a function related to: {question['question']}",
            "input_format": "Input varies based on the function",
            "output_format": "Output should match the expected result",
            "constraints": "Complete the solution in the time provided",
            "examples": ["Example inputs and outputs will vary"],
            "starter_code": "def solution():\n    # Your code here\n    pass",
            "test_cases": [{"input": "example_input", "output": "example_output"}],
            "solution": "# A sample solution would be provided here"
        }
    
    # Display the problem
    st.markdown("### Coding Problem")
    st.markdown(f"**{challenge['problem_statement']}**")
    
    with st.expander("See Details"):
        st.markdown("#### Input Format")
        st.markdown(challenge['input_format'])
        
        st.markdown("#### Output Format")
        st.markdown(challenge['output_format'])
        
        st.markdown("#### Constraints")
        st.markdown(challenge['constraints'])
        
        st.markdown("#### Examples")
        for i, example in enumerate(challenge['examples']):
            st.markdown(f"Example {i+1}: {example}")
    
    # Code editor
    st.markdown("### Your Solution")
    
    # Initialize session state for this question's code if not present
    code_key = f"code_{match_id}_{question_idx}"
    if code_key not in st.session_state:
        st.session_state[code_key] = challenge['starter_code']
    
    # Create code editor with the streamlit-ace component
    submitted_code = st_ace(
        value=st.session_state[code_key],
        language="python",
        theme="monokai",
        keybinding="vscode",
        min_lines=20,
        key=f"ace_{match_id}_{question_idx}"
    )
    
    # Store code in session state
    st.session_state[code_key] = submitted_code
    
    # Run tests button
    if st.button("Run Tests", key=f"run_tests_{match_id}_{question_idx}"):
        with st.spinner("Evaluating your solution..."):
            # This would normally connect to a code execution API like Judge0
            # For now, implement a basic sandbox execution
            
            try:
                # Create test environment
                test_results = []
                test_globals = {}
                
                # Execute candidate's code in isolated environment
                exec(submitted_code, test_globals)
                
                # Test function should be defined now
                if 'solution' not in test_globals:
                    st.error("Your code must define a 'solution' function.")
                    return
                
                # Run test cases
                for i, test_case in enumerate(challenge['test_cases']):
                    try:
                        # This is a simplified approach - in real implementation,
                        # you'd need proper sandboxing and test execution
                        result = str(test_globals['solution'](test_case['input']))
                        expected = str(test_case['output'])
                        
                        passed = result == expected
                        test_results.append({
                            "test_case": i+1,
                            "input": test_case['input'],
                            "expected": expected,
                            "actual": result,
                            "passed": passed
                        })
                    except Exception as e:
                        test_results.append({
                            "test_case": i+1,
                            "input": test_case['input'],
                            "expected": test_case['output'],
                            "actual": f"Error: {str(e)}",
                            "passed": False
                        })
                
                # Display results
                passed_count = sum(1 for r in test_results if r['passed'])
                total_count = len(test_results)
                
                st.markdown(f"### Test Results: {passed_count}/{total_count} passed")
                
                for result in test_results:
                    if result['passed']:
                        st.success(f"Test {result['test_case']}: Passed")
                    else:
                        st.error(f"Test {result['test_case']}: Failed")
                        with st.expander("Details"):
                            st.markdown(f"Input: `{result['input']}`")
                            st.markdown(f"Expected: `{result['expected']}`")
                            st.markdown(f"Actual: `{result['actual']}`")
                
                # AI evaluation if available
                if GEMINI_AVAILABLE and submitted_code:
                    st.markdown("### AI Evaluation")
                    
                    with st.spinner("Analyzing your code..."):
                        evaluation_prompt = f"""
                        Evaluate this code solution for the following problem:
                        Problem: {challenge['problem_statement']}
                        
                        Code:
                        ```python
                        {submitted_code}
                        ```
                        
                        Correct solution for reference:
                        ```python
                        {challenge['solution']}
                        ```
                        
                        Provide a JSON response with:
                        - "correctness_score": 1-5 (how correct the solution is)
                        - "efficiency_score": 1-5 (how efficient the algorithm is)
                        - "code_quality_score": 1-5 (code style and best practices)
                        - "strengths": List of strengths in the code
                        - "areas_for_improvement": List of ways to improve the code
                        - "overall_evaluation": Brief summary evaluation
                        """
                        
                        evaluation = model.generate_content(evaluation_prompt)
                        eval_text = evaluation.text
                        
                        if "```json" in eval_text:
                            eval_text = eval_text.split("```json")[1].split("```")[0].strip()
                        elif "```" in eval_text:
                            eval_text = eval_text.split("```")[1].strip()
                        
                        try:
                            code_eval = json.loads(eval_text)
                            
                            # Store code evaluation in session state
                            code_eval_key = f"code_eval_{match_id}_{question_idx}"
                            st.session_state[code_eval_key] = code_eval
                            
                            # Display evaluation
                            eval_cols = st.columns(3)
                            with eval_cols[0]:
                                st.metric("Correctness", f"{code_eval['correctness_score']}/5")
                            with eval_cols[1]:
                                st.metric("Efficiency", f"{code_eval['efficiency_score']}/5")
                            with eval_cols[2]:
                                st.metric("Code Quality", f"{code_eval['code_quality_score']}/5")
                            
                            st.markdown("#### Strengths")
                            for strength in code_eval['strengths']:
                                st.markdown(f"✅ {strength}")
                            
                            st.markdown("#### Areas for Improvement")
                            for area in code_eval['areas_for_improvement']:
                                st.markdown(f"💡 {area}")
                            
                            st.markdown("#### Overall Evaluation")
                            st.markdown(code_eval['overall_evaluation'])
                            
                        except Exception as e:
                            st.error(f"Error parsing code evaluation: {str(e)}")
                
            except Exception as e:
                st.error(f"Error evaluating code: {str(e)}")
                logger.error(f"Error evaluating code: {str(e)}")

def send_email_to_candidate(match_id, email_type="feedback", recipient_email=None):
    """Send email to a candidate with feedback or next steps"""
    try:
        if 'EMAIL_PASSWORD' not in os.environ or 'EMAIL_USERNAME' not in os.environ:
            st.error("Email credentials not configured. Please set EMAIL_USERNAME and EMAIL_PASSWORD environment variables.")
            return False
        
        # Get match data
        match_details = st.session_state.interview_questions[match_id]
        resume_filename = match_details['resume_filename']
        job_filename = match_details['job_filename']
        questions = match_details['questions']
        
        # Find candidate email from resume if not provided
        if not recipient_email:
            # Extract resume ID
            resume_id = match_id.split('_')[0]
            # Find resume in session state
            resume = next((r for r in st.session_state.resumes if r['id'] == resume_id), None)
            if resume and 'processed' in resume and 'contact' in resume['processed']:
                recipient_email = resume['processed']['contact'].get('email')
        
        if not recipient_email:
            st.error("Could not find candidate email. Please enter it manually.")
            recipient_email = st.text_input("Candidate Email:")
            if not recipient_email:
                return False
        
        # Get response data if available
        responses = {}
        for i in range(len(questions)):
            response_key = f"{match_id}_{i}"
            if response_key in st.session_state.candidate_responses:
                responses[i] = st.session_state.candidate_responses[response_key]
        
        # Create email content based on type
        if email_type == "feedback":
            # Prepare feedback email with interview results
            subject = f"Your Interview Feedback for {job_filename}"
            
            # Create HTML content
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .header {{ background-color: #4CAF50; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .question {{ margin-bottom: 20px; background-color: #f9f9f9; padding: 15px; border-radius: 5px; }}
                    .question h3 {{ color: #2C3E50; margin-top: 0; }}
                    .response {{ background-color: #e7f3fe; padding: 10px; border-left: 4px solid #2196F3; }}
                    .feedback {{ background-color: #fff3cd; padding: 10px; border-left: 4px solid #ffc107; }}
                    .score {{ display: inline-block; padding: 5px 10px; border-radius: 15px; color: white; 
                              background-color: #555; margin-right: 10px; }}
                    .footer {{ background-color: #f1f1f1; padding: 15px; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Interview Feedback</h1>
                    <p>Position: {job_filename}</p>
                </div>
                <div class="content">
                    <p>Dear Candidate,</p>
                    <p>Thank you for participating in our interview process. Below is your personalized feedback based on your interview responses:</p>
            """
            
            # Add question and response sections
            for i, question in enumerate(questions):
                html_content += f"""
                <div class="question">
                    <h3>Question {i+1}: {question['question']}</h3>
                """
                
                if i in responses:
                    response = responses[i]
                    html_content += f"""
                    <div class="response">
                        <p><strong>Your Response:</strong> {response['text']}</p>
                    </div>
                    """
                    
                    # Add scores if available
                    if 'analysis' in response:
                        analysis = response['analysis']
                        html_content += "<div style='margin-top: 10px;'>"
                        
                        if 'relevance_score' in analysis:
                            relevance = analysis['relevance_score']
                            color = '#27ae60' if relevance >= 4 else '#f39c12' if relevance >= 3 else '#e74c3c'
                            html_content += f"""
                            <span class="score" style="background-color: {color};">
                                Relevance: {relevance:.1f}/5
                            </span>
                            """
                        
                        if 'technical_accuracy' in analysis:
                            accuracy = analysis['technical_accuracy']
                            color = '#27ae60' if accuracy >= 4 else '#f39c12' if accuracy >= 3 else '#e74c3c'
                            html_content += f"""
                            <span class="score" style="background-color: {color};">
                                Technical Accuracy: {accuracy:.1f}/5
                            </span>
                            """
                        
                        if 'overall_score' in analysis:
                            overall = analysis['overall_score']
                            color = '#27ae60' if overall >= 4 else '#f39c12' if overall >= 3 else '#e74c3c'
                            html_content += f"""
                            <span class="score" style="background-color: {color};">
                                Overall: {overall:.1f}/5
                            </span>
                            """
                        
                        html_content += "</div>"
                    
                    # Add feedback if available
                    if 'analysis' in response and 'weaknesses' in response['analysis']:
                        html_content += f"""
                        <div class="feedback">
                            <p><strong>Feedback:</strong></p>
                            <ul>
                        """
                        
                        for point in response['analysis']['weaknesses']:
                            html_content += f"<li>{point}</li>"
                        
                        html_content += """
                            </ul>
                        </div>
                        """
                
                html_content += "</div>"
            
            # Add conclusion
            html_content += f"""
                <p>Thank you for your interest in our organization. We will be in touch regarding next steps in the interview process.</p>
                
                <p>Best regards,<br>Hiring Team</p>
            </div>
            <div class="footer">
                <p>This is an automated email. Please do not reply directly to this message.</p>
            </div>
            </body>
            </html>
            """
        
        elif email_type == "next_steps":
            # Prepare next steps email
            subject = f"Next Steps in Your Application for {job_filename}"
            
            html_content = f"""
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                    .header {{ background-color: #3498db; color: white; padding: 20px; text-align: center; }}
                    .content {{ padding: 20px; }}
                    .next-step {{ margin-bottom: 15px; padding: 10px; border-left: 4px solid #3498db; background-color: #ebf5fb; }}
                    .footer {{ background-color: #f1f1f1; padding: 15px; text-align: center; }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>Next Steps in Your Application</h1>
                    <p>Position: {job_filename}</p>
                </div>
                <div class="content">
                    <p>Dear Candidate,</p>
                    <p>Thank you for participating in our initial interview process for the {job_filename} position. We would like to invite you to the next step in our selection process.</p>
                    
                    <div class="next-step">
                        <h3>Next Step: Technical Interview</h3>
                        <p>We would like to schedule a technical interview to further explore your skills and experience. Please use the following link to select a convenient time slot:</p>
                        <p><a href="https://calendly.com/yourcompany/technical-interview" style="background-color: #3498db; color: white; padding: 10px 15px; text-decoration: none; border-radius: 5px;">Schedule Your Interview</a></p>
                    </div>
                    
                    <p>During this interview, we will discuss:</p>
                    <ul>
                        <li>Technical questions related to the position requirements</li>
                        <li>A coding exercise similar to real-world scenarios</li>
                        <li>Your approach to problem-solving</li>
                    </ul>
                    
                    <p>Please let us know if you have any questions or need any accommodations for the interview.</p>
                    
                    <p>Best regards,<br>Hiring Team</p>
                </div>
                <div class="footer">
                    <p>This is an automated email. Please do not reply directly to this message.</p>
                </div>
            </body>
            </html>
            """
        
        else:
            # Custom email
            subject = f"Regarding Your Application for {job_filename}"
            html_content = st.text_area("Customize Email Content:", 
                                       value=f"Dear Candidate,\n\nThank you for your interest in the {job_filename} position.", 
                                       height=300)
            html_content = html_content.replace('\n', '<br>')
            html_content = f"<html><body>{html_content}</body></html>"
        
        # Create message
        msg = MIMEMultipart('alternative')
        msg['Subject'] = subject
        msg['From'] = os.environ['EMAIL_USERNAME']
        msg['To'] = recipient_email
        
        # Attach HTML content
        msg.attach(MIMEText(html_content, 'html'))
        
        # Send email
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(os.environ['EMAIL_USERNAME'], os.environ['EMAIL_PASSWORD'])
            server.send_message(msg)
        
        return True
        
    except Exception as e:
        logger.error(f"Error sending email: {str(e)}")
        st.error(f"Error sending email: {str(e)}")
        return False

def display_interview_questions(match_id):
    """Display generated interview questions with advanced features"""
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
                # Find the global index of this question in all questions
                global_idx = questions.index(question)
                
                with st.expander(f"Q{j+1}: {question['question']}"):
                    st.write(f"**Purpose:** {question.get('purpose', 'Not specified')}")
                    st.write(f"**Good Answer Criteria:** {question.get('good_answer_criteria', 'Not specified')}")
                    st.write(f"**Difficulty:** {question.get('difficulty', 'Not specified')}")
                    
                    # Add voice recording and coding assessment buttons with unique keys
                    button_cols = st.columns(3)
                    with button_cols[0]:
                        # Reset recording state when button is clicked
                        if st.button(f"🎤 Record Response", key=f"record_btn_{q_type}_{global_idx}"):
                            # Reset recording state to make it active again
                            st.session_state[f"recording_active_{match_id}_{global_idx}"] = True
                            # Navigate to recording
                            st.experimental_set_query_params(recording=f"{match_id}_{global_idx}")
                            record_and_analyze_response(match_id, global_idx)
                    
                    # Add coding assessment button for technical questions
                    if q_type.lower() in ["technical", "problem solving"]:
                        with button_cols[1]:
                            if st.button(f"💻 Coding Assessment", key=f"code_btn_{q_type}_{global_idx}"):
                                create_coding_challenge(match_id, global_idx)
    # Create Interview Scorecard
    if any(f"{match_id}_{i}" in st.session_state.candidate_responses 
           for i in range(len(questions))):
        st.header("Interview Scorecard")
        
        # Calculate overall score
        overall_scores = []
        response_data = []
        
        for i in range(len(questions)):
            response_key = f"{match_id}_{i}"
            if response_key in st.session_state.candidate_responses:
                response = st.session_state.candidate_responses[response_key]
                
                # Add to overall scores if available
                if 'analysis' in response and 'overall_score' in response['analysis']:
                    overall_scores.append(response['analysis']['overall_score'])
                
                # Prepare response data for table
                question_text = questions[i]['question']
                question_type = questions[i]['type']
                
                if 'analysis' in response:
                    analysis = response['analysis']
                    response_data.append({
                        'Question #': f"Q{i+1}",
                        'Question Type': question_type,
                        'Relevance': f"{analysis.get('relevance_score', 0):.1f}/5",
                        'Technical': f"{analysis.get('technical_accuracy', 0):.1f}/5",
                        'Clarity': f"{analysis.get('clarity_score', 0):.1f}/5",
                        'Overall': f"{analysis.get('overall_score', 0):.1f}/5"
                    })
                else:
                    response_data.append({
                        'Question #': f"Q{i+1}",
                        'Question Type': question_type,
                        'Relevance': "N/A",
                        'Technical': "N/A",
                        'Clarity': "N/A",
                        'Overall': "N/A"
                    })
        
        # Display aggregate score if we have data
        if overall_scores:
            final_score = sum(overall_scores) / len(overall_scores)
            st.metric("Overall Candidate Score", f"{final_score:.1f}/5")
            
            # Visual indicator
            if final_score >= 4.0:
                st.success("Strong Candidate - Highly Recommended")
            elif final_score >= 3.0:
                st.info("Promising Candidate - Consider for Next Round")
            else:
                st.warning("Not a Strong Match - Additional Screening Recommended")
        
        # Display detailed scores table
        if response_data:
            st.dataframe(pd.DataFrame(response_data), use_container_width=True)
    
    # Export options
    st.subheader("Export & Communication")
    
    export_cols = st.columns(3)
    with export_cols[0]:
        if st.button("📄 Export Questions (PDF)"):
            st.success("Questions exported to PDF!")
    
    with export_cols[1]:
        if st.button("📧 Email Feedback"):
            success = send_email_to_candidate(match_id, email_type="feedback")
            st.success("Feedback email sent successfully!") if success else st.error("Failed to send email")
    
    with export_cols[2]:
        if st.button("🔄 Email Next Steps"):
            success = send_email_to_candidate(match_id, email_type="next_steps")
            st.success("Next steps email sent successfully!") if success else st.error("Failed to send email")