import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter, defaultdict
import os
import sys
import nltk
from datetime import datetime
from nltk.corpus import stopwords

# Add project root to path if needed
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.utils.logger import get_logger
from src.document_processor.chunker import TextChunker
from src.document_processor.preprocessing import TextPreprocessor

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

logger = get_logger(__name__)

def render():
    st.title("Job Matching Analytics")
    
    # Check if documents are uploaded
    if not st.session_state.get("resumes") and not st.session_state.get("job_descriptions"):
        st.warning("Please upload resumes and job descriptions first.")
        st.info("Go to the Upload page to add documents.")
        return
    
    # Create tabs for different analytics views
    tab1, tab2, tab3, tab4 = st.tabs(["Resume Analytics", "Job Analytics", "Match Analytics", "Skill Gap Analysis"])
    
    with tab1:
        if not st.session_state.get("resumes"):
            st.info("No resumes uploaded yet.")
        else:
            resume_analytics()
    
    with tab2:
        if not st.session_state.get("job_descriptions"):
            st.info("No job descriptions uploaded yet.")
        else:
            job_analytics()
    
    with tab3:
        if not st.session_state.get("resumes") or not st.session_state.get("job_descriptions"):
            st.info("Both resumes and job descriptions are needed for match analytics.")
        else:
            match_analytics()
    
    with tab4:
        if not st.session_state.get("resumes") or not st.session_state.get("job_descriptions"):
            st.info("Both resumes and job descriptions are needed for skill gap analysis.")
        else:
            skill_gap_analysis()

def resume_analytics():
    st.header("Resume Collection Analytics")
    
    resumes = st.session_state.resumes
    
    # Top metrics
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Resumes", len(resumes))
    
    with col2:
        avg_length = sum(len(r['original_text']) for r in resumes) / len(resumes) if resumes else 0
        st.metric("Average Length", f"{int(avg_length)} chars")
    
    with col3:
        entities_per_resume = [sum(len(entities) for entities in r['processed'].get('entities', {}).values()) for r in resumes]
        avg_entities = sum(entities_per_resume) / len(entities_per_resume) if entities_per_resume else 0
        st.metric("Avg Entities/Resume", f"{avg_entities:.1f}")
    
    # Skills distribution
    st.subheader("Skills Distribution")
    
    # Extract all skills from all resumes
    all_skills = []
    for resume in resumes:
        if 'entities' in resume['processed'] and 'SKILL' in resume['processed']['entities']:
            all_skills.extend([skill.lower() for skill in resume['processed']['entities']['SKILL']])
    
    skill_counts = Counter(all_skills)
    
    # Display top N skills
    top_n = st.slider("Number of top skills to display", 5, 30, 15)
    
    if skill_counts:
        top_skills = skill_counts.most_common(top_n)
        
        # Create a DataFrame for visualization
        df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
        
        # Create horizontal bar chart
        fig = px.bar(df, x='Count', y='Skill', orientation='h',
                     title=f"Top {top_n} Skills Across All Resumes",
                     color='Count', color_continuous_scale='Viridis')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No skills detected across resumes.")
    
    # Entity Type Distribution
    st.subheader("Entity Type Distribution")
    
    entity_type_counts = defaultdict(int)
    
    for resume in resumes:
        if 'entities' in resume['processed']:
            for entity_type, entities in resume['processed']['entities'].items():
                entity_type_counts[entity_type] += len(entities)
    
    if entity_type_counts:
        # Create pie chart
        fig = px.pie(
            values=list(entity_type_counts.values()),
            names=list(entity_type_counts.keys()),
            title="Distribution of Entity Types Across Resumes"
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No entity data available.")
    
    # Length distribution
    st.subheader("Resume Length Distribution")
    
    lengths = [len(resume['original_text']) for resume in resumes]
    
    if lengths:
        # Create histogram
        fig = px.histogram(
            x=lengths,
            nbins=10,
            title="Resume Length Distribution (characters)",
            labels={'x': 'Length (characters)'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Word cloud option
    st.subheader("Word Cloud")
    
    if st.checkbox("Generate Word Cloud of Common Terms"):
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            # Combine all resume text
            all_text = " ".join([r['processed'].get('clean_text', '') for r in resumes])
            
            # Generate word cloud
            stop_words = set(stopwords.words('english'))
            wordcloud = WordCloud(width=800, height=400, 
                                background_color='white', 
                                stopwords=stop_words, 
                                max_words=100).generate(all_text)
            
            # Display word cloud
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
        except ImportError:
            st.warning("WordCloud package not installed. Run `pip install wordcloud` to use this feature.")
        except Exception as e:
            st.error(f"Error generating word cloud: {str(e)}")

def job_analytics():
    st.header("Job Description Analytics")
    
    jobs = st.session_state.job_descriptions
    
    # Top metrics
    st.subheader("Overview")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Job Descriptions", len(jobs))
    
    with col2:
        avg_length = sum(len(j['original_text']) for j in jobs) / len(jobs) if jobs else 0
        st.metric("Average Length", f"{int(avg_length)} chars")
    
    with col3:
        entities_per_job = [sum(len(entities) for entities in j['processed'].get('entities', {}).values()) for j in jobs]
        avg_entities = sum(entities_per_job) / len(entities_per_job) if entities_per_job else 0
        st.metric("Avg Entities/Job", f"{avg_entities:.1f}")
    
    # Required skills distribution
    st.subheader("Required Skills Distribution")
    
    # Extract all skills from all job descriptions
    all_skills = []
    for job in jobs:
        if 'entities' in job['processed'] and 'SKILL' in job['processed']['entities']:
            all_skills.extend([skill.lower() for skill in job['processed']['entities']['SKILL']])
    
    skill_counts = Counter(all_skills)
    
    # Display top N skills
    top_n = st.slider("Number of top required skills to display", 5, 30, 15, key="job_top_n")
    
    if skill_counts:
        top_skills = skill_counts.most_common(top_n)
        
        # Create a DataFrame for visualization
        df = pd.DataFrame(top_skills, columns=['Skill', 'Count'])
        
        # Create horizontal bar chart
        fig = px.bar(df, x='Count', y='Skill', orientation='h',
                     title=f"Top {top_n} Required Skills Across All Jobs",
                     color='Count', color_continuous_scale='Viridis')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No skills detected across job descriptions.")
    
    # Keyword frequency analysis
    st.subheader("Job Keyword Analysis")
    
    # Get all tokens from job descriptions
    all_tokens = []
    for job in jobs:
        if 'clean_text' in job['processed']:
            tokens = job['processed']['clean_text'].split()
            # Filter out stopwords and short tokens
            stop_words = set(stopwords.words('english'))
            filtered_tokens = [token.lower() for token in tokens if token.lower() not in stop_words and len(token) > 2]
            all_tokens.extend(filtered_tokens)
    
    token_counts = Counter(all_tokens)
    
    if token_counts:
        top_tokens = token_counts.most_common(20)
        
        # Create a DataFrame for visualization
        df = pd.DataFrame(top_tokens, columns=['Keyword', 'Frequency'])
        
        # Create bar chart
        fig = px.bar(df, x='Keyword', y='Frequency', 
                     title="Top 20 Keywords Across All Job Descriptions",
                     color='Frequency', color_continuous_scale='Viridis')
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No keywords available for analysis.")
    
    # Section distribution analysis
    st.subheader("Job Section Analysis")
    
    try:
        chunker = TextChunker()
        
        # Count sections across all job descriptions
        section_counts = defaultdict(int)
        
        for job in jobs:
            sections = chunker.chunk_by_section(job['original_text'])
            for section_title in sections.keys():
                section_counts[section_title] += 1
        
        if section_counts:
            # Create a DataFrame for visualization
            section_items = sorted(section_counts.items(), key=lambda x: x[1], reverse=True)
            df = pd.DataFrame(section_items, columns=['Section', 'Count'])
            
            # Create horizontal bar chart
            fig = px.bar(df, x='Count', y='Section', orientation='h',
                        title="Common Sections in Job Descriptions",
                        color='Count', color_continuous_scale='Viridis')
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No common sections detected across job descriptions.")
    except Exception as e:
        logger.error(f"Error analyzing job sections: {str(e)}")
        st.error("Could not analyze job sections.")

def match_analytics():
    st.header("Match Analytics")
    
    # Check if we have match results in session state
    if not st.session_state.get("match_results"):
        st.info("No match results available yet. Go to the Match page to perform matching first.")
        return
    
    match_results = st.session_state.get("match_results", [])
    
    # Overall match score distribution
    st.subheader("Match Score Distribution")
    
    # Extract match scores from results
    match_scores = [match['score'] for match in match_results]
    
    if match_scores:
        # Create histogram
        fig = px.histogram(
            x=match_scores,
            nbins=10,
            title="Distribution of Match Scores",
            labels={'x': 'Match Score'},
            range_x=[0, 1]
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Match quality metrics
        st.subheader("Match Quality Metrics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Average Match Score", f"{sum(match_scores) / len(match_scores):.2f}")
        
        with col2:
            st.metric("Median Match Score", f"{np.median(match_scores):.2f}")
        
        with col3:
            good_matches = len([score for score in match_scores if score >= 0.7])
            st.metric("Strong Matches (≥0.7)", good_matches)
        
        # Top matches table
        st.subheader("Top Matches")
        
        # Create a table of top matches
        top_matches = sorted(match_results, key=lambda x: x['score'], reverse=True)[:10]
        
        # Create a DataFrame for the table
        data = []
        for match in top_matches:
            data.append({
                "Resume": match['resume_filename'],
                "Job Description": match['job_filename'],
                "Match Score": f"{match['score']:.2f}",
                "Semantic Score": f"{match['details']['semantic_similarity']:.2f}",
                "Keyword Score": f"{match['details']['keyword_similarity']:.2f}"
            })
        
        df = pd.DataFrame(data)
        st.dataframe(df)
        
        # Word cloud of common keywords
        st.subheader("Common Keywords Across Top Matches")
        
        try:
            from wordcloud import WordCloud
            import matplotlib.pyplot as plt
            
            # Extract common keywords from top matches
            common_keywords = []
            for match in top_matches:
                if 'common_keywords' in match['details']:
                    common_keywords.extend(match['details']['common_keywords'])
            
            if common_keywords:
                # Generate word cloud
                keyword_text = " ".join(common_keywords)
                
                wordcloud = WordCloud(
                    width=800, 
                    height=400,
                    background_color='white',
                    max_words=100
                ).generate(keyword_text)
                
                # Display word cloud
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.imshow(wordcloud, interpolation='bilinear')
                ax.axis('off')
                ax.set_title('Common Keywords in Top Matches')
                st.pyplot(fig)
            else:
                st.info("No common keywords found in top matches.")
                
        except ImportError:
            st.warning("WordCloud package not installed. Run `pip install wordcloud` to use this feature.")
        except Exception as e:
            st.error(f"Error generating keyword cloud: {str(e)}")
    else:
        st.info("No match scores available for analysis.")

def skill_gap_analysis():
    st.header("Skill Gap Analysis")
    
    resumes = st.session_state.resumes
    jobs = st.session_state.job_descriptions
    
    # Extract all skills from resumes
    resume_skills = set()
    for resume in resumes:
        if 'entities' in resume['processed'] and 'SKILL' in resume['processed']['entities']:
            resume_skills.update([skill.lower() for skill in resume['processed']['entities']['SKILL']])
    
    # Extract all skills from job descriptions
    job_skills = set()
    for job in jobs:
        if 'entities' in job['processed'] and 'SKILL' in job['processed']['entities']:
            job_skills.update([skill.lower() for skill in job['processed']['entities']['SKILL']])
    
    # Find skill gaps
    skill_gaps = job_skills - resume_skills
    
    # Find skills present in resumes but not in jobs
    extra_skills = resume_skills - job_skills
    
    # Find common skills
    common_skills = resume_skills.intersection(job_skills)
    
    # Create a Venn diagram of skills
    st.subheader("Skills Distribution")
    
    if job_skills or resume_skills:
        col1, col2 = st.columns([3, 2])
        
        with col1:
            # Create set sizes for Venn diagram
            job_only = len(skill_gaps)
            resume_only = len(extra_skills)
            both = len(common_skills)
            
            # Create a Venn diagram using plotly
            fig = go.Figure()
            
            # Add trace for the Venn diagram
            fig.add_trace(go.Scatter(
                x=[0, 1, 0.5],
                y=[0, 0, 0.87],
                mode='text',
                text=[f'Job Skills<br>{job_only + both}', f'Resume Skills<br>{resume_only + both}', f'Common<br>{both}'],
                textfont=dict(
                    family='Arial',
                    size=14,
                    color='black'
                )
            ))
            
            # Add circles
            theta = np.linspace(0, 2*np.pi, 100)
            r = 0.5
            
            # Left circle (Job Skills)
            x1 = 0 + r * np.cos(theta)
            y1 = 0 + r * np.sin(theta)
            
            # Right circle (Resume Skills)
            x2 = 1 + r * np.cos(theta)
            y2 = 0 + r * np.sin(theta)
            
            fig.add_trace(go.Scatter(
                x=x1, y=y1,
                mode='lines',
                fill='toself',
                fillcolor='rgba(31, 119, 180, 0.5)',
                line=dict(color='blue'),
                name='Job Skills'
            ))
            
            fig.add_trace(go.Scatter(
                x=x2, y=y2,
                mode='lines',
                fill='toself',
                fillcolor='rgba(255, 127, 14, 0.5)',
                line=dict(color='orange'),
                name='Resume Skills'
            ))
            
            # Update layout
            fig.update_layout(
                title='Skills Venn Diagram',
                showlegend=True,
                width=600,
                height=400,
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
            )
            
            st.plotly_chart(fig)
        
        with col2:
            st.metric("Total Job Skills", len(job_skills))
            st.metric("Total Resume Skills", len(resume_skills))
            st.metric("Common Skills", len(common_skills))
            st.metric("Skill Gaps", len(skill_gaps))
    
    # Display skill gaps
    st.subheader("Skill Gap Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Skills required by jobs but missing from resumes:**")
        if skill_gaps:
            for skill in sorted(skill_gaps):
                st.write(f"- {skill}")
        else:
            st.info("No skill gaps found.")
    
    with col2:
        st.write("**Skills present in resumes but not required by jobs:**")
        if extra_skills:
            for skill in sorted(extra_skills):
                st.write(f"- {skill}")
        else:
            st.info("No extra skills found.")
    
    # Common skills analysis
    st.subheader("Common Skills Analysis")
    
    if common_skills:
        # Count occurrences of common skills in resumes and jobs
        common_skills_data = []
        
        for skill in common_skills:
            # Count in resumes
            resume_count = sum(1 for resume in resumes 
                              if 'entities' in resume['processed'] 
                              and 'SKILL' in resume['processed']['entities'] 
                              and skill.lower() in [s.lower() for s in resume['processed']['entities']['SKILL']])
            
            # Count in jobs
            job_count = sum(1 for job in jobs 
                           if 'entities' in job['processed'] 
                           and 'SKILL' in job['processed']['entities'] 
                           and skill.lower() in [s.lower() for s in job['processed']['entities']['SKILL']])
            
            common_skills_data.append({
                'Skill': skill,
                'In Resumes': resume_count,
                'In Jobs': job_count,
                'Resume %': resume_count / len(resumes) * 100,
                'Job %': job_count / len(jobs) * 100
            })
        
        # Sort by job demand
        common_skills_data.sort(key=lambda x: x['Job %'], reverse=True)
        
        # Create DataFrame
        df = pd.DataFrame(common_skills_data)
        
        # Display top common skills
        top_n = min(15, len(common_skills_data))
        
        # Create a grouped bar chart
        fig = go.Figure(data=[
            go.Bar(name='In Resumes (%)', x=df['Skill'][:top_n], y=df['Resume %'][:top_n]),
            go.Bar(name='In Jobs (%)', x=df['Skill'][:top_n], y=df['Job %'][:top_n])
        ])
        
        # Update layout
        fig.update_layout(
            title=f'Top {top_n} Common Skills: Demand vs Availability',
            xaxis_title='Skill',
            yaxis_title='Percentage (%)',
            barmode='group'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Display as table too
        st.write("Common Skills Details:")
        st.dataframe(df.head(top_n))
    else:
        st.info("No common skills found between resumes and job descriptions.")

    # Recommendations based on skill gap analysis
    st.subheader("Recommendations")
    
    if skill_gaps:
        st.write("Based on the skill gap analysis, consider focusing on developing these top in-demand skills:")
        
        # Get top 5 most frequently occurring skills in job descriptions that are missing from resumes
        job_skill_freq = defaultdict(int)
        for job in jobs:
            if 'entities' in job['processed'] and 'SKILL' in job['processed']['entities']:
                for skill in job['processed']['entities']['SKILL']:
                    if skill.lower() in skill_gaps:
                        job_skill_freq[skill.lower()] += 1
        
        top_gap_skills = sorted(job_skill_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (skill, count) in enumerate(top_gap_skills, 1):
            demand_percentage = count / len(jobs) * 100
            st.write(f"{i}. **{skill}** - Appears in {count} job descriptions ({demand_percentage:.1f}% of jobs)")
    else:
        st.success("Great! All skills required by the job descriptions are present in at least one resume.")

    # Export option
    st.subheader("Export Analysis")
    
    # Create report data
    report_data = {
        "timestamp": datetime.now().isoformat(),
        "total_resumes": len(resumes),
        "total_jobs": len(jobs),
        "skill_stats": {
            "job_skills_count": len(job_skills),
            "resume_skills_count": len(resume_skills),
            "common_skills_count": len(common_skills),
            "skill_gaps_count": len(skill_gaps)
        },
        "skill_gaps": list(skill_gaps),
        "extra_skills": list(extra_skills),
        "common_skills": list(common_skills)
    }
    
    # Convert to JSON for download
    import json
    report_json = json.dumps(report_data, indent=2)
    
    st.download_button(
        label="Download Skill Gap Analysis (JSON)",
        data=report_json,
        file_name=f"skill_gap_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )