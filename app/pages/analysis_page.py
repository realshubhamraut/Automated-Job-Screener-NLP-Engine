import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
from streamlit_pdf_viewer import pdf_viewer
import logging
from typing import Dict, List, Any
import matplotlib.pyplot as plt

from app.components import charts, document_view

# Initialize basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Make sure NLTK data is downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Define text analysis functions directly in this file to avoid import errors
def analyze_text(text: str) -> Dict[str, Any]:
    """
    Analyze text and extract various statistics
    
    Args:
        text: Text to analyze
        
    Returns:
        Dict with analysis results
    """
    if not text:
        return {
            'word_count': 0,
            'unique_words': 0,
            'sentence_count': 0,
            'avg_word_length': 0,
            'word_freq': {}
        }
    
    # Lowercase and tokenize
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Remove stopwords
    try:
        stop_words = set(stopwords.words('english'))
        filtered_words = [word for word in words if word not in stop_words and len(word) > 2]
    except:
        # If nltk data not available
        filtered_words = [word for word in words if len(word) > 2]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Count sentences
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Calculate average word length
    avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
    
    return {
        'word_count': len(words),
        'unique_words': len(set(words)),
        'sentence_count': len(sentences),
        'avg_word_length': avg_word_length,
        'word_freq': dict(word_freq.most_common(50))
    }

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Extract named entities from text using simple rules-based approach
    
    Args:
        text: Text to extract entities from
        
    Returns:
        Dict with entity categories
    """
    # Super simple entity extraction using regex patterns
    # This is just a placeholder; a real implementation would use NER
    entities = {
        'organizations': [],
        'dates': [],
        'skills': [],
        'education': []
    }
    
    # Simple date pattern
    date_pattern = r'\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}\b|\b\d{4} - (?:Present|Current|\d{4})\b|\b\d{1,2}/\d{1,2}/\d{2,4}\b'
    entities['dates'] = list(set(re.findall(date_pattern, text)))
    
    # Education keywords
    edu_pattern = r'\b(?:Bachelor|Master|PhD|MBA|BSc|MSc|BA|BS|MS|MD|JD|Degree|University|College|School)\s+\w+\b'
    entities['education'] = list(set(re.findall(edu_pattern, text, re.IGNORECASE)))
    
    # Get skills using the find_skills function
    entities['skills'] = find_skills(text)
    
    # Organization patterns (companies often end with these)
    org_pattern = r'\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*(?:\s+(?:Inc|LLC|Ltd|Limited|Corp|Corporation|Co|Company))?\.?\b'
    potential_orgs = re.findall(org_pattern, text)
    # Filter to only longer names
    entities['organizations'] = [org for org in potential_orgs if len(org) > 5][:10]
    
    return entities

def find_skills(text: str) -> List[str]:
    """
    Extract skills from text using a predefined list
    
    Args:
        text: Text to extract skills from
        
    Returns:
        List of skills found
    """
    # Expanded skills list
    extended_skills = [
        # Programming Languages
        "Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "Swift",
        "TypeScript", "Go", "Kotlin", "R", "Rust", "Scala", "Perl", "HTML", "CSS",
        "Objective-C", "Elixir", "Haskell", "Clojure", "MATLAB", "Bash/Shell", "Lua",

        # Frameworks & Libraries
        "React", "Angular", "Vue", "Django", "Flask", "Spring", "Node.js", 
        "Express", "TensorFlow", "PyTorch", "Keras", "Pandas", "NumPy", "Scikit-learn",
        "jQuery", "Bootstrap", "Laravel", "ASP.NET", "Ruby on Rails",
        "Ember.js", "Backbone.js", "Svelte", "Meteor", "Redux", "GraphQL", "Tailwind CSS",

        # Databases
        "SQL", "MySQL", "PostgreSQL", "MongoDB", "Oracle", "SQLite", "Redis",
        "Cassandra", "DynamoDB", "MariaDB", "Elasticsearch", "Firebase", "Neo4j",
        "InfluxDB", "CockroachDB", "CouchDB",

        # Cloud & DevOps
        "AWS", "Azure", "Google Cloud", "Docker", "Kubernetes", "Jenkins",
        "Terraform", "Ansible", "CircleCI", "GitHub Actions", "Prometheus", "Grafana",
        "Heroku", "Serverless", "Git", "Linux", "CI/CD", "Nginx", "Apache", "Vagrant",
        "DevSecOps", "Chef", "Puppet",

        # Management & Methodologies
        "Agile", "Scrum", "Kanban", "JIRA", "Confluence", "Project Management", 
        "Product Management", "Lean", "Six Sigma", "Waterfall", "PMP",
        "Risk Management", "Change Management", "Stakeholder Management",
        "Scrum Master", "Time Management", "Resource Management", "Budgeting", "Conflict Resolution",

        # General Professional Skills
        "Leadership", "Team Management", "Strategic Planning", "Public Speaking",
        "Negotiation", "Data Analysis", "Problem Solving", "Critical Thinking",
        "Communication", "Teamwork", "Adaptability", "Creativity", "Emotional Intelligence",
        "Interpersonal Skills", "Networking", "Collaboration", "Mentoring",

        # Design & UX
        "UI/UX", "Figma", "Adobe XD", "Sketch", "Photoshop", "Illustrator",
        "InDesign", "User Research", "Wireframing", "Prototyping", "InVision",
        "Axure", "Zeplin", "Miro", "Canva", "Responsive Design", "Accessibility", "Design Thinking",

        # Business & Analytics
        "Excel", "PowerPoint", "Tableau", "Power BI", "Data Visualization",
        "Business Intelligence", "Financial Analysis", "Market Research",
        "SEO", "Digital Marketing", "CRM", "ERP", "Business Process Management",
        "Data Governance", "ETL", "Big Data Analytics", "RPA",

        # AI & Machine Learning
        "Machine Learning", "Deep Learning", "NLP", "Computer Vision",
        "Neural Networks", "Reinforcement Learning", "Data Mining",
        "Data Engineering", "Apache Spark", "Hadoop", "Bayesian Modeling", "Generative AI",

        # Mobile Development
        "iOS", "Android", "React Native", "Flutter", "Xamarin", "SwiftUI",
        "Kotlin Multiplatform", "Ionic", "Cordova", "NativeScript", "Android Studio", "Gradle",

        # Testing
        "Unit Testing", "Selenium", "Cypress", "JUnit", "TestNG", "Mocha", 
        "Jest", "Pytest", "Cucumber", "Robot Framework", "Postman", "Load Testing",
        "Performance Testing", "Security Testing", "Test Automation",

        # Additional Emerging & Specialized Skills
        "Cybersecurity", "Ethical Hacking", "Penetration Testing", "Information Security",
        "Blockchain", "Smart Contracts", "Solidity", "Cryptography", "IoT", "AR", "VR",
        "Unity", "Unreal Engine", "Game Development", "3D Modeling", "Animation",
        "Software Architecture", "Microservices", "Design Patterns", "System Design",
        "API Design", "RESTful Services", "GraphQL APIs", "Containerization", "Virtualization",
        "Networking Fundamentals", "Continuous Integration", "Continuous Delivery",
        "Cloud Security", "GDPR Compliance", "Data Privacy"
    ]
    
    found_skills = []
    text_lower = text.lower()
    
    for skill in extended_skills:
        # Simple string comparison approach - more reliable than regex for special characters
        skill_lower = skill.lower()
        
        # Use a simpler approach to avoid regex errors
        if skill_lower in text_lower:
            # For single words, check if they are standalone words
            if len(skill_lower.split()) == 1:
                # Check if it's a standalone word by looking at surrounding characters
                for match in re.finditer(re.escape(skill_lower), text_lower):
                    start, end = match.span()
                    
                    # Check if the match is at the beginning or preceded by non-word character
                    is_word_start = (start == 0 or not text_lower[start-1].isalnum())
                    
                    # Check if the match is at the end or followed by non-word character
                    is_word_end = (end == len(text_lower) or not text_lower[end].isalnum())
                    
                    if is_word_start and is_word_end:
                        found_skills.append(skill)
                        break
            else:
                # For phrases, direct containment is reasonable
                found_skills.append(skill)
    
    return list(set(found_skills))

def get_ats_compatibility_score(resume_text: str, job_text: str) -> Dict[str, Any]:
    """
    Calculate ATS compatibility score between a resume and job description
    
    Args:
        resume_text: The resume text
        job_text: The job description text
        
    Returns:
        Dictionary with score and details
    """
    # Extract keywords from job description
    job_keywords = set(re.findall(r'\b[A-Za-z][A-Za-z+#\.-]*[A-Za-z0-9]\b', job_text.lower()))
    # Remove common words and short words
    try:
        stop_words = set(stopwords.words('english'))
        job_keywords = {word for word in job_keywords if word not in stop_words and len(word) > 2}
    except:
        job_keywords = {word for word in job_keywords if len(word) > 2}
    
    # Extract words from resume
    resume_words = set(re.findall(r'\b[A-Za-z][A-Za-z+#\.-]*[A-Za-z0-9]\b', resume_text.lower()))
    
    # Find common keywords
    common_keywords = job_keywords.intersection(resume_words)
    
    # Calculate keyword match percentage
    keyword_score = len(common_keywords) / len(job_keywords) if job_keywords else 0
    
    # Get formatting and structure score (simplified)
    structure_score = 0.85  # Simplified placeholder
    
    # Calculate final ATS score (weighted average)
    ats_score = 0.7 * keyword_score + 0.3 * structure_score
    
    return {
        'ats_score': ats_score,
        'keyword_match': keyword_score,
        'structure_score': structure_score,
        'common_keywords': list(common_keywords),
        'missing_keywords': list(job_keywords - common_keywords)
    }


def render():
    st.title("Analyze Shortlisted Candidates")
    
    # Check if shortlisted candidates exist
    if not st.session_state.get('shortlisted_candidates', []):
        st.warning("No candidates have been shortlisted yet. Please go to the Match tab to shortlist candidates.")
        
        # Check if any resumes are available for demo purposes
        if st.session_state.get('resumes', []):
            if st.button("View all resumes anyway"):
                analyze_resumes(st.session_state.resumes)
        return
    
    # Display shortlisted candidates
    st.success(f"Analyzing {len(st.session_state.shortlisted_candidates)} shortlisted candidates")
    
    # Sort shortlisted candidates by match score (highest first)
    shortlisted = sorted(
        st.session_state.shortlisted_candidates, 
        key=lambda x: x.get('match_score', 0), 
        reverse=True
    )
    
    # Analyze shortlisted resumes
    analyze_resumes(shortlisted)

def analyze_resumes(resumes):
    """Analyze resume content and display insights"""
    
    # Create sidebar for document selection
    st.sidebar.subheader("Filter Candidates")
    
    selected_doc_idx = st.sidebar.selectbox(
        "Choose a candidate to analyze",
        options=list(range(len(resumes))),
        format_func=lambda x: f"{resumes[x]['filename']} {get_score_display(resumes[x])}",
        key="doc_select"
    )
    
    selected_doc = resumes[selected_doc_idx]
    
    # Find the matching job description if available
    matching_job = None
    if 'matched_job_id' in selected_doc:
        for job in st.session_state.job_descriptions:
            if job['id'] == selected_doc['matched_job_id']:
                matching_job = job
                break
    
    # Main analysis content
    st.header(f"Candidate Analysis: {selected_doc['filename']}")
    
    # Create main layout with tabs
    main_tabs = st.tabs(["Overview", "Skills Analysis", "ATS Compatibility", "Resume Details"])
    
    # OVERVIEW TAB
    with main_tabs[0]:
        # Top section with PDF preview and metrics
        col1, col2 = st.columns([1, 1])
        
        with col1:
            # PDF preview in a card-like container
            st.subheader("Resume Preview")
            
            if 'filepath' in selected_doc and selected_doc['filepath'].lower().endswith('.pdf'):
                with st.container():
                    st.markdown("""
                    <style>
                    .pdf-container { 
                        border: 1px solid #ddd; 
                        border-radius: 5px; 
                        padding: 10px;
                        background-color: #f9f9f9;
                    }
                    </style>
                    """, unsafe_allow_html=True)
                    
                    with st.expander("Click to view resume", expanded=True):
                        try:
                            pdf_viewer(selected_doc['filepath'], width=350, height=500)
                        except Exception as e:
                            st.error(f"Could not load PDF: {str(e)}")
            else:
                st.info("No PDF available for this candidate")
                
        with col2:
            # Score metrics and key stats
            st.subheader("Key Metrics")
            
            # Match score with gauge chart
            if 'match_score' in selected_doc:
                st.markdown("#### Match Score")
                # Try to use the charts module's gauge function if available
                try:
                    charts.render_gauge(selected_doc['match_score'], "Match Score")
                except:
                    # Fallback to simple metric if gauge chart fails
                    st.metric("Match Score", f"{selected_doc['match_score']:.2f}")
            
            # ATS Score
            if matching_job:
                ats_results = get_ats_compatibility_score(
                    selected_doc['processed']['clean_text'],
                    matching_job['processed']['clean_text']
                )
                
                st.markdown("#### ATS Compatibility Score")
                try:
                    charts.render_gauge(ats_results['ats_score'], "ATS Score")
                except:
                    st.metric("ATS Score", f"{ats_results['ats_score']:.2f}")
                    
                # Show keyword match percentage
                st.metric("Keyword Match", f"{ats_results['keyword_match']:.1%}")
            
            # Document statistics
            text_analysis = analyze_text(selected_doc['processed']['clean_text'])
            metrics_col1, metrics_col2 = st.columns(2)
            with metrics_col1:
                st.metric("Word Count", text_analysis['word_count'])
                st.metric("Unique Words", text_analysis['unique_words'])
            with metrics_col2:
                st.metric("Sentences", text_analysis['sentence_count'])
                st.metric("Avg. Word Length", f"{text_analysis['avg_word_length']:.1f}")
        
        # Common Tags Section
        st.subheader("Top Common Tags with Job Description")
        if matching_job and 'match_details' in selected_doc:
            # Show common tags between resume and job description
            common_tags = selected_doc['match_details'].get('common_keywords', [])
            if common_tags:
                # Create tag pills with CSS
                st.markdown("""
                <style>
                .tag-pill {
                    display: inline-block;
                    background-color: #e3f2fd;
                    color: #1565c0;
                    border-radius: 16px;
                    padding: 6px 12px;
                    margin: 4px;
                    font-size: 14px;
                    font-weight: 500;
                }
                .tag-container {
                    display: flex;
                    flex-wrap: wrap;
                    gap: 8px;
                    margin-bottom: 20px;
                }
                </style>
                """, unsafe_allow_html=True)
                
                tag_html = '<div class="tag-container">'
                for tag in common_tags[:15]:  # Show top 15 tags
                    tag_html += f'<div class="tag-pill">{tag}</div>'
                tag_html += '</div>'
                
                st.markdown(tag_html, unsafe_allow_html=True)
            else:
                st.info("No common tags found with the job description")
        else:
            st.info("No matching job description found for tag comparison")
            
        # Key insights cards
        st.subheader("Key Insights")
        
        insight_col1, insight_col2, insight_col3 = st.columns(3)
        
        with insight_col1:
            st.markdown("""
            <div style="border:1px solid #ddd; border-radius:5px; padding:15px; background-color:#f8f9fa;">
                <h4 style="color:#1565c0;">Skills Match</h4>
                <p>The candidate's skills match rate with job requirements.</p>
                <h2 style="color:#2e7d32; text-align:center;">
            """, unsafe_allow_html=True)
            
            # Calculate skills match if we have a job description
            if matching_job and 'match_details' in selected_doc:
                skills_match = selected_doc['match_details'].get('keyword_similarity', 0)
                st.markdown(f"<h2 style='text-align:center;'>{skills_match:.0%}</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='text-align:center;'>N/A</h2>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
            
        with insight_col2:
            st.markdown("""
            <div style="border:1px solid #ddd; border-radius:5px; padding:15px; background-color:#f8f9fa;">
                <h4 style="color:#1565c0;">Experience Level</h4>
                <p>Estimated years of experience based on resume content.</p>
            """, unsafe_allow_html=True)
            
            # Simple heuristic for experience estimation
            text = selected_doc['processed']['clean_text']
            experience_indicators = len(re.findall(r'\b\d{4}\b', text))  # Count years mentioned
            
            if experience_indicators > 10:
                experience = "Senior (5+ years)"
            elif experience_indicators > 5:
                experience = "Mid-level (2-5 years)"
            else:
                experience = "Entry-level (0-2 years)"
                
            st.markdown(f"<h2 style='text-align:center;'>{experience}</h2>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with insight_col3:
            st.markdown("""
            <div style="border:1px solid #ddd; border-radius:5px; padding:15px; background-color:#f8f9fa;">
                <h4 style="color:#1565c0;">Education</h4>
                <p>Highest education level detected in resume.</p>
            """, unsafe_allow_html=True)
            
            # Extract education
            entities = extract_entities(selected_doc['processed']['clean_text'])
            education = entities.get('education', [])
            
            if education:
                # Try to identify highest education level
                if any(re.search(r'phd|doctorate|doctor', edu, re.I) for edu in education):
                    highest_edu = "PhD/Doctorate"
                elif any(re.search(r'master|mba|ms\b|ma\b', edu, re.I) for edu in education):
                    highest_edu = "Master's Degree"
                elif any(re.search(r'bachelor|bs\b|ba\b|btech', edu, re.I) for edu in education):
                    highest_edu = "Bachelor's Degree"
                else:
                    highest_edu = "Other"
                    
                st.markdown(f"<h2 style='text-align:center;'>{highest_edu}</h2>", unsafe_allow_html=True)
            else:
                st.markdown("<h2 style='text-align:center;'>Not Specified</h2>", unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)
    
    # SKILLS ANALYSIS TAB
    with main_tabs[1]:
        st.subheader("Skills & Expertise")
        
        # Extract skills
        skills = find_skills(selected_doc['processed']['clean_text'])
        
        if skills:
            # Group skills by category
            skill_categories = {
                "Programming": ["Python", "Java", "JavaScript", "C++", "C#", "Ruby", "PHP", "TypeScript"],
                "Web Dev": ["React", "Angular", "Vue", "HTML", "CSS", "Node.js", "Bootstrap", "jQuery"],
                "Data": ["SQL", "PostgreSQL", "MySQL", "MongoDB", "Data Analysis", "Big Data"],
                "Cloud & DevOps": ["AWS", "Azure", "Docker", "Kubernetes", "CI/CD", "Git"],
                "AI & ML": ["Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "NLP"],
                "Business": ["Project Management", "Agile", "Scrum", "Leadership"]
            }
            
            # Categorize skills
            categorized = {cat: [] for cat in skill_categories}
            categorized["Other"] = []
            
            for skill in skills:
                categorized_flag = False
                for category, category_skills in skill_categories.items():
                    if any(cat_skill.lower() in skill.lower() for cat_skill in category_skills):
                        categorized[category].append(skill)
                        categorized_flag = True
                        break
                        
                if not categorized_flag:
                    categorized["Other"].append(skill)
            
            # Display skills by category in columns
            categories = list(categorized.keys())
            col_width = 3
            num_rows = (len(categories) + col_width - 1) // col_width
            
            for row in range(num_rows):
                cols = st.columns(col_width)
                for i, col in enumerate(cols):
                    idx = row * col_width + i
                    if idx < len(categories):
                        category = categories[idx]
                        if categorized[category]:
                            with col:
                                st.markdown(f"##### {category}")
                                for skill in categorized[category]:
                                    st.markdown(f"- {skill}")
            
            # Skills visualization
            st.subheader("Skills Visualization")
            try:
                # Create skill frequency data
                skill_freq = {}
                text_lower = selected_doc['processed']['clean_text'].lower()
                
                for skill in skills:
                    skill_lower = skill.lower()
                    count = len(re.findall(r'\b' + re.escape(skill_lower) + r'\b', text_lower))
                    skill_freq[skill] = count if count > 0 else 1
                    
                # Use top skills for visualization
                top_skills = dict(sorted(skill_freq.items(), key=lambda x: x[1], reverse=True)[:10])
                
                # Try different visualizations
                viz_col1, viz_col2 = st.columns([1, 1])
                
                with viz_col1:
                    try:
                        charts.render_bar_chart(
                            list(top_skills.keys()), 
                            list(top_skills.values()),
                            "Top Skills by Frequency", 
                            "Skill", 
                            "Mentions"
                        )
                    except:
                        st.dataframe(pd.DataFrame({"Skill": top_skills.keys(), "Count": top_skills.values()}))
                
                with viz_col2:
                    try:
                        charts.render_wordcloud(" ".join(skills * 3))  # Repeat skills to make wordcloud more varied
                    except:
                        st.write("Skill cloud visualization not available")
            except Exception as e:
                st.error(f"Error generating skill visualizations: {str(e)}")
        else:
            st.info("No skills detected in the resume. Consider adding relevant skills keywords.")
    
    # ATS COMPATIBILITY TAB
    with main_tabs[2]:
        st.subheader("ATS Compatibility Analysis")
        
        if matching_job:
            ats_results = get_ats_compatibility_score(
                selected_doc['processed']['clean_text'],
                matching_job['processed']['clean_text']
            )
            
            # ATS Score card
            st.markdown("""
            <style>
            .ats-card {
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 20px;
                text-align: center;
                background: linear-gradient(135deg, #f5f7fa 0%, #e4e8ec 100%);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            </style>
            """, unsafe_allow_html=True)
            
            score_color = "#4caf50" if ats_results['ats_score'] > 0.7 else "#ff9800" if ats_results['ats_score'] > 0.5 else "#f44336"
            
            st.markdown(f"""
            <div class="ats-card">
                <h3>ATS Compatibility Score</h3>
                <h1 style="font-size: 3rem; color:{score_color};">{ats_results['ats_score']:.0%}</h1>
                <p>This is how well the resume matches ATS requirements for the job description.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Score breakdown
            score_col1, score_col2 = st.columns(2)
            
            with score_col1:
                try:
                    charts.render_gauge(ats_results['keyword_match'], "Keyword Match")
                except:
                    st.metric("Keyword Match Score", f"{ats_results['keyword_match']:.1%}")
            
            with score_col2:
                try:
                    charts.render_gauge(ats_results['structure_score'], "Format & Structure")
                except:
                    st.metric("Format & Structure Score", f"{ats_results['structure_score']:.1%}")
            
            # Keywords analysis
            st.subheader("Keyword Analysis")
            keyword_col1, keyword_col2 = st.columns(2)
            
            with keyword_col1:
                st.markdown("##### ✅ Matching Keywords")
                if ats_results['common_keywords']:
                    for kw in ats_results['common_keywords'][:15]:
                        st.markdown(f"- {kw}")
                else:
                    st.info("No matching keywords found")
            
            with keyword_col2:
                st.markdown("##### ❌ Missing Keywords")
                if ats_results['missing_keywords']:
                    for kw in ats_results['missing_keywords'][:15]:
                        st.markdown(f"- {kw}")
                else:
                    st.success("No missing keywords!")
                    
    # RESUME DETAILS TAB
    with main_tabs[3]:
        # Display extracted entities
        entities = extract_entities(selected_doc['processed']['clean_text'])
        
        entity_tabs = st.tabs(["Organizations", "Dates", "Education"])
        
        with entity_tabs[0]:
            st.subheader("Organizations Mentioned")
            if entities.get('organizations', []):
                for org in entities['organizations']:
                    st.markdown(f"• {org}")
            else:
                st.write("No organizations detected.")
        
        with entity_tabs[1]:
            st.subheader("Timeline & Dates")
            if entities.get('dates', []):
                for date in entities['dates']:
                    st.markdown(f"• {date}")
            else:
                st.write("No dates detected.")
        
        with entity_tabs[2]:
            st.subheader("Education")
            if entities.get('education', []):
                for edu in entities['education']:
                    st.markdown(f"• {edu}")
            else:
                st.write("No education details detected.")
        
        # Raw text display
        st.subheader("Full Resume Text")
        with st.expander("Show/Hide Full Text"):
            st.text_area("", selected_doc['processed']['clean_text'], height=400)

def get_score_display(resume):
    """Format the score display for the sidebar selectbox"""
    if 'match_score' in resume:
        return f"(Score: {resume['match_score']:.2f})"
    return ""