import streamlit as st
import pandas as pd
import time
import json
import os
from typing import Dict, List, Any, Optional
import google.generativeai as genai
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

def render():
    """Render the interview questions page"""
    st.title("AI Interview Question Generator")
    st.write("Generate personalized interview questions based on resume and job description matches.")
    
    # Check for required session state data
    if not st.session_state.resumes:
        st.warning("Please upload at least one resume first.")
        return
    
    if not st.session_state.job_descriptions:
        st.warning("Please upload at least one job description first.")
        return
    
    if not st.session_state.get('match_results'):
        st.warning("Please match resumes to job descriptions first to generate interview questions.")
        st.info("Go to the 'Match' tab to perform matching.")
        return
    
    # Initialize session state for questions if not exists
    if 'interview_questions' not in st.session_state:
        st.session_state.interview_questions = {}
    
    # Sidebar settings
    with st.sidebar:
        st.subheader("Question Generation Settings")
        
        num_questions = st.slider(
            "Number of Questions", 
            min_value=3, 
            max_value=15, 
            value=8, 
            step=1,
            help="Number of interview questions to generate"
        )
        
        question_types = st.multiselect(
            "Question Types",
            options=["Technical", "Behavioral", "Experience", "Problem Solving", "Role-specific", "Culture Fit"],
            default=["Technical", "Experience", "Problem Solving"],
            help="Types of questions to generate"
        )
        
        difficulty = st.select_slider(
            "Question Difficulty",
            options=["Basic", "Intermediate", "Advanced", "Expert"],
            value="Intermediate",
            help="Difficulty level of generated questions"
        )
        
        if GEMINI_AVAILABLE:
            generation_model = st.radio(
                "Generation Model",
                options=["Gemini Pro", "Rule-based"],
                index=0,
                help="Choose the model for generating questions"
            )
        else:
            st.error("Gemini API not available. Using rule-based generation.")
            generation_model = "Rule-based"
    
    # Main content
    st.header("Generate Interview Questions")
    
    # Match selection
    st.subheader("Select a Resume-Job Match")
    
    # Create dataframe for match results
    if st.session_state.match_results:
        match_df = pd.DataFrame([
            {
                'Resume': result['resume_filename'],
                'Job': result['job_filename'],
                'Score': f"{result['score']:.2f}",
                'ID': f"{result['resume_id']}_{result['job_id']}"
            } for result in st.session_state.match_results
        ])
        
        # Sort by score
        match_df['Numeric_Score'] = [float(result['score']) for result in st.session_state.match_results]
        match_df = match_df.sort_values('Numeric_Score', ascending=False).drop(columns=['Numeric_Score'])
        
        # Display match results
        st.dataframe(match_df.drop(columns=['ID']), use_container_width=True)
        
        # Match selection
        selected_match = st.selectbox(
            "Select a match for interview question generation",
            options=match_df['ID'].tolist(),
            format_func=lambda x: f"{match_df[match_df['ID']==x]['Resume'].iloc[0]} - {match_df[match_df['ID']==x]['Job'].iloc[0]} (Score: {match_df[match_df['ID']==x]['Score'].iloc[0]})"
        )
        
        # Find the selected match details
        resume_id, job_id = selected_match.split('_', 1)
        
        selected_resume = None
        selected_job = None
        selected_match_details = None
        
        # Find the resume and job objects
        for resume in st.session_state.resumes:
            if resume['id'] == resume_id:
                selected_resume = resume
                break
        
        for job in st.session_state.job_descriptions:
            if job['id'] == job_id:
                selected_job = job
                break
                
        # Find the match result details
        for match in st.session_state.match_results:
            if match['resume_id'] == resume_id and match['job_id'] == job_id:
                selected_match_details = match
                break
        
        if selected_resume and selected_job and selected_match_details:
            # Display selected resume and job description
            col1, col2 = st.columns(2)
            
            with col1:
                document_view.render_document_card(
                    selected_resume['filename'], 
                    selected_resume['processed']['clean_text'][:500] + "...", 
                    "Resume"
                )
            
            with col2:
                document_view.render_document_card(
                    selected_job['filename'], 
                    selected_job['processed']['clean_text'][:500] + "...", 
                    "Job Description"
                )
            
            # Generate questions button - HIGHLIGHTED
            st.markdown("### 👇 Click the button below to generate questions 👇")
            if st.button("Generate Interview Questions", key="gen_questions_button"):
                match_id = f"{resume_id}_{job_id}"
                
                with st.spinner("Generating personalized interview questions..."):
                    try:
                        start_time = time.time()
                        
                        # Get resume and job text
                        resume_text = selected_resume['processed']['clean_text']
                        job_text = selected_job['processed']['clean_text']
                        
                        # Generate questions
                        if generation_model == "Gemini Pro" and GEMINI_AVAILABLE:
                            questions = generate_questions_gemini(
                                resume_text, 
                                job_text, 
                                num_questions, 
                                question_types, 
                                difficulty
                            )
                        else:
                            # Fallback to rule-based generation
                            question_gen = QuestionGenerator()
                            questions = question_gen.generate_questions(
                                resume_text,
                                job_text,
                                num_questions=num_questions,
                                question_types=question_types,
                                difficulty=difficulty.lower()
                            )
                        
                        # Store generated questions in session state
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
                        
                        end_time = time.time()
                        st.success(f"Generated {len(questions)} interview questions in {end_time - start_time:.2f} seconds")
                    
                    except Exception as e:
                        logger.error(f"Error generating questions: {str(e)}")
                        st.error(f"Error generating questions: {str(e)}")
            
            # Display generated questions if available
            match_id = f"{resume_id}_{job_id}"
            if match_id in st.session_state.interview_questions:
                display_interview_questions(st.session_state.interview_questions[match_id])
        else:
            # If objects not found, show warning but still allow generating questions
            st.warning("Could not find matching resume and job details. This may be due to ID mismatches between your selected match and the stored data.")
            
            # Get resume and job filenames from the match_df
            resume_filename = match_df[match_df['ID']==selected_match]['Resume'].iloc[0] 
            job_filename = match_df[match_df['ID']==selected_match]['Job'].iloc[0]
            
            st.info(f"You can still generate questions for {resume_filename} and {job_filename}, but they may be less personalized.")
            
            # Find the resume and job by filename instead of ID
            backup_resume = None
            backup_job = None
            
            for resume in st.session_state.resumes:
                if resume['filename'] == resume_filename:
                    backup_resume = resume
                    break
                    
            for job in st.session_state.job_descriptions:
                if job['filename'] == job_filename:
                    backup_job = job
                    break
            
            # Display selected resume and job description
            if backup_resume and backup_job:
                col1, col2 = st.columns(2)
                
                with col1:
                    document_view.render_document_card(
                        backup_resume['filename'], 
                        backup_resume['processed']['clean_text'][:500] + "...", 
                        "Resume"
                    )
                
                with col2:
                    document_view.render_document_card(
                        backup_job['filename'], 
                        backup_job['processed']['clean_text'][:500] + "...", 
                        "Job Description"
                    )
            
            # Alternative button for generating questions anyway
            st.markdown("### 👇 Click the button below to generate questions anyway 👇")
            if st.button("Generate Interview Questions Anyway", key="gen_questions_anyway_button"):
                match_id = selected_match  # Use the selected match ID directly
                
                with st.spinner("Generating interview questions..."):
                    try:
                        start_time = time.time()
                        
                        if backup_resume and backup_job:
                            # Get text content
                            resume_text = backup_resume['processed']['clean_text']
                            job_text = backup_job['processed']['clean_text']
                            
                            # Generate questions
                            if generation_model == "Gemini Pro" and GEMINI_AVAILABLE:
                                questions = generate_questions_gemini(
                                    resume_text, 
                                    job_text, 
                                    num_questions, 
                                    question_types, 
                                    difficulty
                                )
                            else:
                                # Fallback to rule-based generation
                                question_gen = QuestionGenerator()
                                questions = question_gen.generate_questions(
                                    resume_text,
                                    job_text,
                                    num_questions=num_questions,
                                    question_types=question_types,
                                    difficulty=difficulty.lower()
                                )
                            
                            # Store generated questions in session state
                            st.session_state.interview_questions[match_id] = {
                                'questions': questions,
                                'resume_filename': resume_filename,
                                'job_filename': job_filename,
                                'generation_time': time.time(),
                                'settings': {
                                    'num_questions': num_questions,
                                    'question_types': question_types,
                                    'difficulty': difficulty,
                                    'model': generation_model
                                }
                            }
                            
                            end_time = time.time()
                            st.success(f"Generated {len(questions)} interview questions in {end_time - start_time:.2f} seconds")
                        else:
                            st.error("Could not find the documents by filename. Please try matching again.")
                    
                    except Exception as e:
                        logger.error(f"Error generating questions: {str(e)}")
                        st.error(f"Error generating questions: {str(e)}")
            
            # Display generated questions if available
            if selected_match in st.session_state.interview_questions:
                display_interview_questions(st.session_state.interview_questions[selected_match])


def generate_questions_gemini(
    resume_text: str, 
    job_text: str, 
    num_questions: int, 
    question_types: List[str], 
    difficulty: str
) -> List[Dict[str, str]]:
    """
    Generate interview questions using Google's Gemini API
    
    Args:
        resume_text: Resume text
        job_text: Job description text
        num_questions: Number of questions to generate
        question_types: Types of questions to generate
        difficulty: Difficulty level of questions
        
    Returns:
        List of question dictionaries
    """
    try:
        # Prepare the prompt
        prompt = f"""
        You are a professional hiring manager preparing for an interview. Create {num_questions} unique interview questions based on the following:
        
        Resume:
        {resume_text[:3000]}  # Limiting to 3000 chars to avoid exceeding token limits
        
        Job Description:
        {job_text[:3000]}  # Limiting to 3000 chars to avoid exceeding token limits
        
        Generate {num_questions} {difficulty.lower()} level questions of the following types: {', '.join(question_types)}.
        
        For each question:
        1. Consider the specific skills and experience in the resume that match the job description
        2. Focus on assessing both technical capability and problem-solving abilities
        3. Include questions that verify the candidate's claimed experience
        
        Format the response as a JSON array with each question having these fields:
        - "question": The interview question
        - "type": The question type (one of the types specified)
        - "difficulty": The difficulty level
        - "purpose": A brief explanation of what this question aims to assess
        - "good_answer_criteria": What would make a good answer to this question

        Only respond with valid JSON.
        """
        
        # Generate response from Gemini
        response = model.generate_content(prompt)
        
        # Extract JSON from response
        json_text = response.text
        
        # Sometimes the response might contain markdown formatting, so let's extract just the JSON part
        if "```json" in json_text:
            json_text = json_text.split("```json")[1].split("```")[0].strip()
        elif "```" in json_text:
            json_text = json_text.split("```")[1].strip()
        
        # Parse JSON into Python objects
        questions = json.loads(json_text)
        
        # Ensure we have the right format
        if isinstance(questions, list) and len(questions) > 0:
            # Validate and fix each question
            for q in questions:
                if "question" not in q:
                    q["question"] = "Please describe your experience with this role."
                if "type" not in q:
                    q["type"] = question_types[0] if question_types else "Technical"
                if "difficulty" not in q:
                    q["difficulty"] = difficulty
                if "purpose" not in q:
                    q["purpose"] = "Assessing candidate qualifications"
                if "good_answer_criteria" not in q:
                    q["good_answer_criteria"] = "Clear, specific answer demonstrating experience"
            
            return questions[:num_questions]  # Ensure we return only the requested number
        else:
            logger.error(f"Invalid response format from Gemini API: {json_text}")
            raise ValueError("Invalid response format from API")
            
    except Exception as e:
        logger.error(f"Error with Gemini API: {str(e)}")
        # Return a fallback set of generic questions
        return generate_fallback_questions(num_questions, question_types, difficulty)

def generate_fallback_questions(num_questions: int, question_types: List[str], difficulty: str) -> List[Dict[str, str]]:
    """Generate fallback questions when API fails"""
    
    fallback_questions = [
        {
            "question": "Can you walk me through your relevant experience for this role?",
            "type": "Experience",
            "difficulty": difficulty,
            "purpose": "To understand the candidate's background",
            "good_answer_criteria": "Clear explanation of relevant experience with specific examples"
        },
        {
            "question": "What technical skills do you bring that make you a good fit for this position?",
            "type": "Technical",
            "difficulty": difficulty,
            "purpose": "To assess technical qualifications",
            "good_answer_criteria": "Specific technical skills that match job requirements"
        },
        {
            "question": "Describe a challenging problem you solved in a previous role.",
            "type": "Problem Solving",
            "difficulty": difficulty,
            "purpose": "To evaluate problem-solving abilities",
            "good_answer_criteria": "Structured approach to problem-solving with measurable results"
        },
        {
            "question": "How do you stay updated with the latest developments in your field?",
            "type": "Professional Development",
            "difficulty": difficulty,
            "purpose": "To assess learning attitude",
            "good_answer_criteria": "Demonstrates continuous learning and professional development"
        },
        {
            "question": "What interests you about this position?",
            "type": "Motivation",
            "difficulty": difficulty,
            "purpose": "To understand candidate's interest",
            "good_answer_criteria": "Shows genuine interest and knowledge about the role"
        }
    ]
    
    # Filter by question types if provided
    if question_types:
        filtered_questions = [q for q in fallback_questions if q["type"] in question_types]
        # If filtering removed all questions, return original set
        if not filtered_questions:
            filtered_questions = fallback_questions
    else:
        filtered_questions = fallback_questions
    
    # Return requested number of questions, repeating if necessary
    result = []
    for i in range(num_questions):
        result.append(filtered_questions[i % len(filtered_questions)])
    
    return result

def display_interview_questions(question_data: Dict[str, Any]):
    """
    Display the generated interview questions
    
    Args:
        question_data: Question data dictionary from session state
    """
    st.header(f"Interview Questions: {question_data['resume_filename']} - {question_data['job_filename']}")
    
    # Display settings
    settings = question_data['settings']
    st.caption(f"Generated using {settings['model']} | Difficulty: {settings['difficulty']} | Types: {', '.join(settings['question_types'])}")
    
    # Group questions by type
    questions_by_type = {}
    for q in question_data['questions']:
        q_type = q.get('type', 'General')
        if q_type not in questions_by_type:
            questions_by_type[q_type] = []
        questions_by_type[q_type].append(q)
    
    # Create tabs for each question type
    if questions_by_type:
        tabs = st.tabs(list(questions_by_type.keys()))
        
        for i, (q_type, tab) in enumerate(zip(questions_by_type.keys(), tabs)):
            with tab:
                for j, question in enumerate(questions_by_type[q_type]):
                    with st.expander(f"Q{j+1}: {question['question']}", expanded=j==0):
                        st.write(f"**Purpose:** {question.get('purpose', 'N/A')}")
                        st.write(f"**Difficulty:** {question.get('difficulty', 'N/A')}")
                        st.write(f"**Good Answer Criteria:**")
                        st.write(question.get('good_answer_criteria', 'N/A'))
    
    # Export options
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export to PDF"):
            st.info("PDF export functionality will be implemented in a future update.")
    
    with col2:
        if st.button("Export to Text"):
            questions_text = generate_questions_text(question_data)
            st.download_button(
                label="Download Text File",
                data=questions_text,
                file_name=f"interview_questions_{question_data['resume_filename']}_{question_data['job_filename']}.txt",
                mime="text/plain"
            )

def generate_questions_text(question_data: Dict[str, Any]) -> str:
    """
    Generate a formatted text version of the questions
    
    Args:
        question_data: Question data dictionary
        
    Returns:
        Formatted text of questions
    """
    text = f"INTERVIEW QUESTIONS\n"
    text += f"Resume: {question_data['resume_filename']}\n"
    text += f"Job: {question_data['job_filename']}\n"
    text += f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(question_data['generation_time']))}\n\n"
    
    # Group by type
    questions_by_type = {}
    for q in question_data['questions']:
        q_type = q.get('type', 'General')
        if q_type not in questions_by_type:
            questions_by_type[q_type] = []
        questions_by_type[q_type].append(q)
    
    # Add questions by type
    for q_type, questions in questions_by_type.items():
        text += f"\n{q_type.upper()} QUESTIONS:\n"
        text += "=" * (len(q_type) + 10) + "\n\n"
        
        for i, q in enumerate(questions):
            text += f"{i+1}. {q['question']}\n"
            text += f"   Purpose: {q.get('purpose', 'N/A')}\n"
            text += f"   Difficulty: {q.get('difficulty', 'N/A')}\n"
            text += f"   Good Answer: {q.get('good_answer_criteria', 'N/A')}\n\n"
    
    return text