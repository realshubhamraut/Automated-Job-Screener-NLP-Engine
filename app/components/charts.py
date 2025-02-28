import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Union
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

def render_gauge(score: float, title: str = "Match Score", min_val: float = 0, max_val: float = 1):
    """
    Render a gauge-like chart showing the score using seaborn
    
    Args:
        score (float): Score value
        title (str): Chart title
        min_val (float): Minimum value for the score
        max_val (float): Maximum value for the score
    """
    # Normalize score to 0-1 range if needed
    normalized_score = score
    if min_val != 0 or max_val != 1:
        normalized_score = (score - min_val) / (max_val - min_val)
        # Ensure score is in 0-1 range
        normalized_score = max(0, min(1, normalized_score))
    
    # Convert to percentage for display
    percentage = normalized_score * 100
    
    # Create a figure with custom dimensions
    fig, ax = plt.subplots(figsize=(4, 3))
    
    # Define color based on score
    if percentage < 50:
        color = 'red'
    elif percentage < 70:
        color = 'orange'
    else:
        color = 'green'
    
    # Create a half-circle gauge-like chart
    sns.set_style("whitegrid")
    
    # Create the background
    background = np.linspace(0, 100, 100)
    background_colors = ['red' if i < 50 else 'orange' if i < 70 else 'green' for i in background]
    
    # Plot the gauge with a semi-circle
    ax.bar(0, 100, width=6, color='lightgray', alpha=0.3)
    ax.bar(0, percentage, width=6, color=color)
    
    # Add a text label for the percentage
    ax.text(0, 50, f"{percentage:.1f}%", ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Configure the plot
    ax.set_ylim(0, 100)
    ax.set_xlim(-3, 3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])
    ax.set_title(title, fontsize=12)
    
    # Render in Streamlit
    st.pyplot(fig)

def create_component_scores_chart(scores: Dict[str, float]) -> go.Figure:
    """
    Create a bar chart for component scores
    
    Args:
        scores (Dict[str, float]): Dictionary of scores
        
    Returns:
        plotly.graph_objects.Figure: Bar chart
    """
    df = pd.DataFrame(list(scores.items()), columns=['Component', 'Score'])
    df['Percentage'] = df['Score'] * 100  # Convert to percentage
    
    fig = px.bar(
        df, 
        x='Component', 
        y='Percentage', 
        text='Percentage',
        color='Percentage',
        color_continuous_scale=[(0, 'red'), (0.5, 'yellow'), (1, 'green')],
        range_color=[0, 100]
    )
    
    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
    
    fig.update_layout(
        title="Component Scores",
        xaxis_title="",
        yaxis_title="Match Percentage",
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_keyword_venn(resume_keywords: List[str], job_keywords: List[str]) -> go.Figure:
    """
    Create a Venn diagram showing keyword overlap
    
    Args:
        resume_keywords (List[str]): Resume keywords
        job_keywords (List[str]): Job keywords
        
    Returns:
        plotly.graph_objects.Figure: Venn diagram
    """
    # Convert lists to sets
    resume_set = set(resume_keywords)
    job_set = set(job_keywords)
    
    # Calculate overlaps
    overlap = resume_set.intersection(job_set)
    missing = job_set - resume_set
    extra = resume_set - job_set
    
    # Create data for plotting
    data = [
        {"Category": "Matching Keywords", "Count": len(overlap)},
        {"Category": "Missing Keywords", "Count": len(missing)},
        {"Category": "Extra Keywords", "Count": len(extra)}
    ]
    
    df = pd.DataFrame(data)
    
    # Create bar chart
    fig = px.bar(
        df,
        x="Category",
        y="Count",
        color="Category",
        color_discrete_map={
            "Matching Keywords": "green",
            "Missing Keywords": "red",
            "Extra Keywords": "blue"
        },
        text="Count"
    )
    
    fig.update_traces(textposition='outside')
    
    fig.update_layout(
        title="Keyword Analysis",
        xaxis_title="",
        yaxis_title="Count",
        height=300,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    
    return fig

def create_wordcloud(text: str, title: str = None) -> plt.Figure:
    """
    Create a word cloud from text
    
    Args:
        text (str): Text to create word cloud from
        title (str, optional): Chart title
        
    Returns:
        matplotlib.figure.Figure: Word cloud figure
    """
    # Create word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        max_words=100,
        contour_width=3,
        contour_color='steelblue'
    ).generate(text)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    
    if title:
        ax.set_title(title)
    
    return fig