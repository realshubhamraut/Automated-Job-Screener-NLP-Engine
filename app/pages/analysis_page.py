import os
import re
import time
import json
import glob
import streamlit as st
from dotenv import load_dotenv
import google.generativeai as genai
from src.utils.logger import get_logger

logger = get_logger(__name__)

# List of technical domains and specific technologies for skill extraction
TECH_DOMAINS = [
    # Programming Languages
    "python", "java", "javascript", "typescript", "c++", "c#", "ruby", "go", "php", "swift",
    "kotlin", "rust", "scala", "perl", "r", "matlab", "dart", "groovy", "bash", "powershell",
    "shell script", "vba", "cobol", "fortran", "lisp", "haskell", "erlang", "clojure", "f#",
    
    # Web Development
    "html", "css", "react", "angular", "vue", "node.js", "express", "django", "flask",
    "spring", "rails", "laravel", "asp.net", "jquery", "bootstrap", "tailwind",
    "webpack", "babel", "sass", "less", "redux", "next.js", "gatsby", "nuxt.js",
    "ember.js", "svelte", "backbone.js", "meteor", "sails.js", "web components",
    
    # Mobile Development
    "android", "ios", "flutter", "react native", "swift", "kotlin", "objective-c",
    "xamarin", "ionic", "cordova", "nativescript", "swiftui", "jetpack compose",
    
    # Databases
    "sql", "mysql", "postgresql", "mongodb", "dynamodb", "firebase", "redis", "cassandra",
    "oracle", "sqlite", "mariadb", "neo4j", "elasticsearch", "couchdb", "mssql",
    "nosql", "graphql", "supabase", "prisma", "sequelize", "mongoose", "jdbc", "odbc",
    
    # DevOps & Cloud
    "aws", "azure", "gcp", "docker", "kubernetes", "jenkins", "terraform", "ansible",
    "circleci", "travis ci", "github actions", "gitlab ci", "puppet", "chef", "prometheus", 
    "grafana", "elk stack", "serverless", "lambda", "s3", "ec2", "rds", "cloudformation",
    "fargate", "eks", "ecs", "beanstalk", "cloudfront", "route53", "vpc", "devops",
    "ci/cd", "continuous integration", "continuous deployment", "microservices",
    
    # Data Science & AI
    "machine learning", "deep learning", "tensorflow", "pytorch", "scikit-learn", "pandas",
    "numpy", "scipy", "keras", "nlp", "computer vision", "data mining", "ai", "artificial intelligence",
    "neural networks", "reinforcement learning", "big data", "hadoop", "spark", "tableau", "power bi",
    "data science", "data engineering", "data analytics", "data visualization", "data pipeline",
    "etl", "data warehouse", "predictive modeling", "statistical analysis", "r studio",
    "jupyter", "databricks", "sagemaker", "mlops", "model training", "model inference",
    
    # Testing & QA
    "selenium", "cypress", "jest", "mocha", "junit", "pytest", "testng", "cucumber", "jmeter",
    "postman", "soapui", "qtest", "test automation", "unit testing", "integration testing",
    "end-to-end testing", "regression testing", "performance testing", "load testing",
    "stress testing", "qa", "quality assurance", "test cases", "test plans", "test strategy",
    
    # Security
    "cybersecurity", "penetration testing", "security", "encryption", "firewall", "oauth",
    "jwt", "authentication", "authorization", "owasp", "encryption", "ssl", "tls", "https",
    "vpn", "identity management", "access control", "soc", "security operations", "compliance",
    "gdpr", "hipaa", "pci", "soc2", "security audit", "threat modeling", "vulnerability assessment",
    
    # Project Management & Methodologies
    "agile", "scrum", "kanban", "waterfall", "jira", "confluence", "trello", "lean", 
    "sdlc", "ci/cd", "devops", "gitflow", "sprint", "backlog", "user story", "epic",
    "product owner", "scrum master", "retrospective", "sprint planning", "daily standups",

    # Systems & Infrastructure
    "linux", "unix", "windows server", "macos", "active directory", "ldap", "dns", "dhcp",
    "networking", "tcp/ip", "http", "https", "ssh", "ftp", "sftp", "load balancer",
    "reverse proxy", "nginx", "apache", "iis", "tomcat", "weblogic", "websphere",
    "system architecture", "high availability", "fault tolerance", "disaster recovery",
    "backup", "restore", "monitoring", "logging", "alerting", "caching"
]

# Load environment variables
load_dotenv()

# Get Google API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Configure the generative AI library
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Cache for storing responses to reduce API calls
if "analysis_cache" not in st.session_state:
    st.session_state.analysis_cache = {}

def load_job_descriptions():
    """Load all processed job description files"""
    job_descriptions = []
    
    # Path to the project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    job_descriptions_dir = os.path.join(project_root, 'data', 'processed', 'job_descriptions')
    
    if not os.path.exists(job_descriptions_dir):
        logger.warning(f"Job descriptions directory not found: {job_descriptions_dir}")
        return job_descriptions
    
    # Find all JSON files in the job descriptions directory
    json_files = glob.glob(os.path.join(job_descriptions_dir, "*.json"))
    
    for file_path in json_files:
        try:
            with open(file_path, 'r') as f:
                job_data = json.load(f)
            
            # Even if the JSON doesn't have document_type, include it if it has filename
            if job_data.get('filename'):
                # If the job data doesn't contain processed text, try to load it
                if not job_data.get('processed', {}).get('clean_text'):
                    # Try to find the original text file
                    base_name = os.path.splitext(job_data['filename'])[0]
                    text_file_path = os.path.join(
                        project_root, 'data', 'raw', 'job_descriptions', 
                        f"{base_name}.txt"
                    )
                    
                    # If text file exists, read its content
                    if os.path.exists(text_file_path):
                        try:
                            with open(text_file_path, 'r', encoding='utf-8') as text_file:
                                text_content = text_file.read()
                                # Create or update the processed field
                                if 'processed' not in job_data:
                                    job_data['processed'] = {}
                                job_data['processed']['clean_text'] = text_content
                        except Exception as e:
                            logger.warning(f"Could not read text file {text_file_path}: {str(e)}")
                
                # Add to our list if it has filename
                job_descriptions.append(job_data)
                logger.info(f"Loaded job description: {job_data.get('filename', 'Unknown')}")
                
        except Exception as e:
            logger.error(f"Error loading job description {file_path}: {str(e)}")
    
    return job_descriptions

def query_gemini(prompt, max_output_tokens=250):
    """Query Google's Gemini 2.0 Flash-Lite model"""
    if not GEMINI_API_KEY:
        return "API key not found. Please add GEMINI_API_KEY to your .env file."
        
    # Check cache first
    cache_key = f"{prompt[:100]}"
    if cache_key in st.session_state.analysis_cache:
        return st.session_state.analysis_cache[cache_key]
    
    try:
        # Configure the model - using Flash-Lite for faster responses and higher rate limits
        model = genai.GenerativeModel('gemini-2.0-flash-lite')
        
        # Set generation config
        generation_config = {
            "temperature": 0.1,
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": max_output_tokens,
        }
        
        # Generate content
        response = model.generate_content(
            prompt,
            generation_config=generation_config
        )
        
        # Get the text from the response
        result = response.text
        
        # Cache the result
        st.session_state.analysis_cache[cache_key] = result
        return result
    except Exception as e:
        logger.error(f"Error querying Gemini: {e}")
        return f"Error: {str(e)}"

def extract_contact_info(resume_text):
    """Extract contact information directly using Gemini"""
    prompt = f"""
    Extract the following information from this resume text:
    - Full name
    - Email address
    - Phone number
    - Location (city/state)
    
    Format your response exactly like this:
    NAME: [full name]
    EMAIL: [email]
    PHONE: [phone]
    LOCATION: [location]
    
    Resume text:
    {resume_text[:3000]}
    
    Only return the formatted extraction, nothing else.
    """
    
    response = query_gemini(prompt)
    
    # Initialize with generic values
    name = "Unknown Candidate" 
    email = "No Email Found"
    phone = "No Phone Found"
    location = "Unknown Location"
    
    # Extract information from the response
    if "NAME:" in response.upper():
        name_match = re.search(r'NAME:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if name_match:
            name = name_match.group(1).strip()
    
    if "EMAIL:" in response.upper():
        email_match = re.search(r'EMAIL:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if email_match:
            email = email_match.group(1).strip()
    
    if "PHONE:" in response.upper():
        phone_match = re.search(r'PHONE:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if phone_match:
            phone = phone_match.group(1).strip()
    
    if "LOCATION:" in response.upper():
        location_match = re.search(r'LOCATION:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
        if location_match:
            location = location_match.group(1).strip()
    
    # Direct email extraction as backup
    if email == "No Email Found" or email.lower() == "not found" or email.lower() == "none":
        email_direct = re.search(r'[\w.+-]+@[\w-]+\.[\w.-]+', resume_text)
        if email_direct:
            email = email_direct.group(0)
    
    return {
        "name": name,
        "email": email,
        "phone": phone,
        "location": location
    }

def extract_job_title(resume_text):
    """Extract current job title directly using Gemini"""
    prompt = f"""
    What is this person's current or most recent job title from the resume?
    Return only the job title, nothing else.
    
    Resume:
    {resume_text[:3000]}
    """
    
    response = query_gemini(prompt, max_output_tokens=100)
    
    # Clean up the response
    job_title = response.strip()
    if not job_title or job_title.lower() in ["unknown", "not specified", "not mentioned", "not found"]:
        job_title = "Not Specified"
        
    return job_title

def extract_years_of_experience(resume_text):
    """Extract years of experience directly using Gemini"""
    prompt = f"""
    How many total years of professional work experience does this person have based on the resume?
    Return only a number representing the years. If unclear, estimate based on work history.
    
    Resume:
    {resume_text[:3000]}
    """
    
    response = query_gemini(prompt, max_output_tokens=50)
    
    # Try to extract just the number
    numbers = re.findall(r'\b\d+\b', response)
    if numbers:
        # Get the first number found
        try:
            years = int(numbers[0])
            return min(years, 50)  # Cap at 50 years
        except:
            pass
    
    # Default if we can't extract a number
    return 5  # Default to reasonable value instead of 0

def extract_technical_skills_from_text(text):
    """Extract technical skills from text using predefined TECH_DOMAINS list"""
    text_lower = text.lower()
    found_skills = []
    
    for skill in TECH_DOMAINS:
        # Look for skill as a whole word to avoid partial matches
        pattern = r'\b' + re.escape(skill) + r'\b'
        if re.search(pattern, text_lower):
            # Use the original skill name from the list
            found_skills.append(skill)
    
    # Remove duplicates while preserving order
    unique_skills = []
    seen = set()
    for skill in found_skills:
        if skill.lower() not in seen:
            seen.add(skill.lower())
            unique_skills.append(skill)
    
    return unique_skills

def extract_skills(text, is_job_description=False):
    """Extract skills from text"""
    if is_job_description:
        # For job descriptions, use Gemini for better context understanding
        prompt = f"""
        Extract all technical skills, tools, programming languages, and technologies required in this job description.
        Format as a comma-separated list.
        Focus on hard skills and technical requirements only.
        
        Job Description:
        {text[:3000]}
        """
        
        response = query_gemini(prompt, max_output_tokens=300)
        
        # Split and clean the response
        skills = []
        if ',' in response:
            skills = [s.strip() for s in response.split(',')]
        elif '\n' in response:
            skills = [s.strip() for s in response.split('\n')]
        else:
            skills = [s.strip() for s in response.split()]
        
        # Filter out empty or too short/long skills
        skills = [skill for skill in skills if 2 < len(skill) < 30]
        
        # Remove duplicates
        unique_skills = []
        seen = set()
        for skill in skills:
            if skill.lower() not in seen:
                seen.add(skill.lower())
                unique_skills.append(skill)
        
        return unique_skills
    else:
        # For resumes, use our predefined technical domains list
        return extract_technical_skills_from_text(text)

def extract_resume_summary(resume_text):
    """Generate a comprehensive professional summary using Gemini"""
    prompt = f"""
    Write a detailed professional summary for this candidate based on their resume.
    Include their background, key skills, experience level, and notable achievements.
    Write at least 3-4 sentences and be specific about their expertise.
    
    Resume:
    {resume_text[:3000]}
    """
    
    response = query_gemini(prompt, max_output_tokens=350)
    
    # If the summary is too short, use a backup approach
    if len(response.split()) < 20:
        job_title = extract_job_title(resume_text)
        skills = extract_skills(resume_text)
        skill_text = ", ".join(skills[:5]) if skills else "various technical skills"
        
        # Create a generic but useful summary
        backup_summary = f"""
        This candidate is a {job_title} professional with expertise in {skill_text}. 
        Their background demonstrates relevant experience for this role, with technical skills 
        that align with the position requirements. Their professional history suggests they could 
        bring valuable insights and capabilities to your team.
        """
        return backup_summary.strip()
    
    return response.strip()

def compare_skills(candidate_skills, job_skills):
    """Compare candidate skills to job skills and find missing ones"""
    # Convert both lists to lowercase for case-insensitive comparison
    candidate_skills_lower = [skill.lower() for skill in candidate_skills]
    job_skills_lower = [skill.lower() for skill in job_skills]
    
    # Find common skills (skills present in both lists)
    common_skills = []
    for i, job_skill in enumerate(job_skills):
        if job_skills_lower[i] in candidate_skills_lower:
            common_skills.append(job_skill)
    
    # Find missing skills (skills in job but not in candidate)
    missing_skills = []
    for i, job_skill in enumerate(job_skills):
        if job_skills_lower[i] not in candidate_skills_lower:
            missing_skills.append(job_skill)
    
    return common_skills, missing_skills

def extract_resume_info(resume_text):
    """Extract all information from the resume using Gemini and predefined skill domains"""
    try:
        # Initialize the structure with better defaults
        info = {
            "contact": {"name": "Unknown Candidate", "email": "No Email Found", "phone": "No Phone Found", "location": "Unknown Location"},
            "job_title": "Not Specified",
            "experience_years": 0,
            "skills": [],
            "summary": "No summary available."
        }
        
        # Extract contact information
        contact_info = extract_contact_info(resume_text)
        # Direct assignment
        info["contact"] = contact_info
        
        # Extract job title
        job_title = extract_job_title(resume_text)
        if job_title and len(job_title.strip()) > 0:
            info["job_title"] = job_title
        
        # Extract experience years
        info["experience_years"] = extract_years_of_experience(resume_text)
        
        # Extract skills using our predefined technical domains
        skills = extract_skills(resume_text)
        if skills:
            info["skills"] = skills
        
        # Generate summary
        summary = extract_resume_summary(resume_text)
        if summary and len(summary.strip()) > 10:
            info["summary"] = summary
        
        return info
        
    except Exception as e:
        logger.error(f"Error extracting resume info: {e}")
        return {
            "error": str(e),
            "contact": {"name": "API Error", "email": "API Error", "phone": "API Error", "location": "API Error"},
            "job_title": "API Error",
            "experience_years": 0,
            "skills": ["API Error"],
            "summary": f"Failed to extract information: {str(e)}"
        }
def render():
    """Single render function for the analysis page"""
    st.title("Candidate Analysis")
    
    if not st.session_state.get('shortlisted_candidates'):
        st.warning("No candidates shortlisted for analysis. Please shortlist candidates first.")
        return

    st.markdown("""
    <style>
    .card {
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #e0e0e0;
        margin-bottom: 0.8rem;
        background-color: white;
    }
    .skill-tag {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        border: 1px solid #90caf9;
        border-radius: 16px;
        padding: 2px 8px;
        margin: 2px;
        font-size: 12px;
    }
    .missing-tag {
        background-color: #ffebee;
        color: #c62828;
        border-color: #ef9a9a;
    }
    .matching-tag {
        background-color: #e8f5e9;
        color: #2e7d32;
        border-color: #a5d6a7;
    }
    .debug-info {
        background-color: #f5f5f5;
        border: 1px solid #ddd;
        padding: 10px;
        margin: 10px 0;
        font-family: monospace;
        font-size: 12px;
        overflow-x: auto;
        white-space: pre-wrap;
    }
    </style>
    """, unsafe_allow_html=True)

    # Check for API token
    if not GEMINI_API_KEY:
        st.error("Google API key not found. Please add it to your .env file as GEMINI_API_KEY.")
        api_key = st.text_input("Enter your Google API key:", type="password", key="api_key_input")
        if api_key:
            os.environ["GEMINI_API_KEY"] = api_key
            st.session_state["GEMINI_API_KEY"] = api_key
            genai.configure(api_key=api_key)
            st.success("API token set for this session! Refreshing...")
            st.experimental_rerun()
        st.markdown("""
        ### How to get a Google API key:
        1. Go to [Google AI Studio](https://makersuite.google.com/)
        2. Create an account if you don't have one
        3. Go to "API keys" in the left sidebar
        4. Create a new API key
        5. Add the key to your .env file as GEMINI_API_KEY=your_key_here
        """)
        return

    
    # Sort candidates by score
    candidates = sorted(
        st.session_state.shortlisted_candidates,
        key=lambda x: x.get('match_score', 0),
        reverse=True
    )
    candidate_options = {f"{c['filename']} ({c.get('match_score', 0):.2f})": i for i, c in enumerate(candidates)}
    selected_key = st.sidebar.selectbox("Select Candidate", 
                                       list(candidate_options.keys()),
                                       key="candidate_selector")
    selected_candidate = candidates[candidate_options[selected_key]]
    
    # Load job descriptions from processed directory
    if "job_descriptions_loaded" not in st.session_state:
        with st.spinner("Loading job descriptions..."):
            st.session_state.job_descriptions_loaded = load_job_descriptions()
            st.session_state.job_descriptions_dict = {
                jd.get('filename', f"Job {i}"): jd 
                for i, jd in enumerate(st.session_state.job_descriptions_loaded)
            }
    
    # Add job description selection from sidebar
    job_description_text = ""
    st.sidebar.subheader("Job Description")
    
    if st.session_state.job_descriptions_dict:
        # Create a list of job description options
        job_options = list(st.session_state.job_descriptions_dict.keys())
        
        # Add a "None" option
        job_options = ["None"] + job_options
        
        # Select job description
        selected_job_name = st.sidebar.selectbox(
            "Select job description for comparison",
            options=job_options,
            key="analysis_job_select"
        )
        
        if selected_job_name != "None":
            # Get the selected job description
            selected_job = st.session_state.job_descriptions_dict[selected_job_name]
            
            # Get the text content
            if selected_job.get('processed', {}).get('clean_text'):
                job_description_text = selected_job['processed']['clean_text']
                st.session_state['job_description_text'] = job_description_text
                st.session_state['selected_job_name'] = selected_job_name
            else:
                st.sidebar.warning("Selected job description has no text content.")
    else:
        st.sidebar.warning("No job descriptions found. Please upload job descriptions first.")
        # Try to use previously set job description
        job_description_text = st.session_state.get('job_description_text', '')
    
    resume_text = selected_candidate.get('processed', {}).get('clean_text', '')
    if not resume_text:
        st.error("No resume text found for this candidate.")
        return
    
    # Print first bit of resume for debug purposes


    with st.spinner("Analyzing resume..."):
        # Extract all information at once
        info = extract_resume_info(resume_text)
        
        # Extract skills from resume and job description
        candidate_skills = info.get("skills", [])
        job_skills = []
        common_skills = []
        missing_skills = []
        
        if job_description_text:
            # Extract skills from job description
            with st.spinner("Extracting skills from job description..."):
                job_skills = extract_skills(job_description_text, is_job_description=True)
                
                # Compare skills to find matches and gaps
                common_skills, missing_skills = compare_skills(candidate_skills, job_skills)
                
        
        # Show extracted info for debugging

        st.subheader(f"Analysis for: {selected_candidate['filename']}")
        
        # Display selected job description name if available
        if job_description_text and st.session_state.get('selected_job_name'):
            st.caption(f"Compared with job: {st.session_state.get('selected_job_name')}")
        
        # Display metrics
        match_score = selected_candidate.get('match_score', 0) * 100
        semantic_score = selected_candidate.get('match_details', {}).get('semantic_similarity', 0) * 100
        keyword_score = selected_candidate.get('match_details', {}).get('keyword_similarity', 0) * 100
        exp_years = info.get("experience_years", 0)

        cols = st.columns(4)
        cols[0].metric("Overall Match", f"{match_score:.1f}%")
        cols[1].metric("Semantic Match", f"{semantic_score:.1f}%")
        cols[2].metric("Keyword Match", f"{keyword_score:.1f}%")
        cols[3].metric("Experience", f"{exp_years} years")

        st.markdown("### Candidate Profile")
        profile_cols = st.columns(2)
        with profile_cols[0]:
            contact = info.get("contact", {})
            contact_html = f'''
            <div class="card">
            <h4>Contact Information</h4>
            <p><strong>Name:</strong> {contact.get('name')}</p>
            <p><strong>Email:</strong> {contact.get('email')}</p>
            <p><strong>Phone:</strong> {contact.get('phone')}</p>
            <p><strong>Location:</strong> {contact.get('location')}</p>
            <p><strong>Current Title:</strong> {info.get('job_title')}</p>
            </div>
            '''
            st.markdown(contact_html, unsafe_allow_html=True)

        with profile_cols[1]:
            summary_text = info.get("summary", "Not available")
            summary_html = f'''
            <div class="card">
            <h4>Resume Summary</h4>
            <p>{summary_text}</p>
            </div>
            '''
            st.markdown(summary_html, unsafe_allow_html=True)

       # Skills Section
        st.markdown("### Skills")
        skills_cols = st.columns(2)
        
        # Display candidate skills found using our technical domain list
        with skills_cols[0]:
            st.markdown("<h4>Candidate Skills</h4>", unsafe_allow_html=True)
            if candidate_skills:
                rendered_skills = " ".join([f'<span class="skill-tag">{s}</span>' for s in candidate_skills])
                st.markdown(rendered_skills, unsafe_allow_html=True)
            else:
                st.write("No specific skills detected.")
        
        # Display missing skills section
        with skills_cols[1]:
            # Emphasize this is a skills gap analysis focused on missing skills
            st.markdown("<h4>Missing Skills</h4>", unsafe_allow_html=True)
            
            if not job_description_text:
                st.warning("⚠️ No job description selected. Please select a job description from the sidebar to see missing skills.")
            elif job_skills:
                # Calculate and display the skill match percentage

                # Display missing skills prominently with red tags
                if missing_skills:
                    rendered_missing = " ".join([f'<span class="skill-tag missing-tag">{s}</span>' for s in missing_skills])
                    st.markdown(rendered_missing, unsafe_allow_html=True)
                    
                    # Add a note explaining what missing skills represent
                    st.markdown("<small>*These skills were requested in the job description but not found in the resume.</small>", unsafe_allow_html=True)
                if len(job_skills) > 0:
                    skill_match_percent = (len(common_skills) / len(job_skills)) * 100
                    st.markdown(f"**Match Rate:** {skill_match_percent:.1f}% ({len(common_skills)} of {len(job_skills)} required skills)")
                
                else:
                    st.success("✅ No skill gaps detected! The candidate has all required skills.")
            else:
                with st.spinner("Analyzing job requirements..."):
                    # Try to extract job skills one more time
                    job_skills = extract_skills(job_description_text, is_job_description=True)
                    if job_skills:
                        # Redo the comparison
                        common_skills, missing_skills = compare_skills(candidate_skills, job_skills)
                        
                        # Calculate and display the skill match percentage
                        if len(job_skills) > 0:
                            skill_match_percent = (len(common_skills) / len(job_skills)) * 100
                            st.markdown(f"**Match Rate:** {skill_match_percent:.1f}% ({len(common_skills)} of {len(job_skills)} required skills)")
                        
                        # Display missing skills with red tags
                        if missing_skills:
                            rendered_missing = " ".join([f'<span class="skill-tag missing-tag">{s}</span>' for s in missing_skills])
                            st.markdown(rendered_missing, unsafe_allow_html=True)
                            
                            # Add a note explaining what missing skills represent
                            st.markdown("<small>*These skills were requested in the job description but not found in the resume.</small>", unsafe_allow_html=True)
                        else:
                            st.success("✅ No skill gaps detected! The candidate has all required skills.")
                    else:
                        st.write("Could not extract specific requirements from the job description.")

        # Add a direct question form
        st.markdown("### Ask Questions About This Resume")
        
        # User-defined question input
        test_question = st.text_input("Enter your question:", key="custom_question")
        
        if test_question:
            with st.spinner("Processing question..."):
                prompt = f"""
                Answer the following question about this candidate's resume:
                
                Question: {test_question}
                
                Resume:
                {resume_text[:3000]}
                
                Provide a detailed and specific answer based only on information in the resume.
                """
                answer = query_gemini(prompt, max_output_tokens=400)
                st.write(f"**Response:** {answer}")
