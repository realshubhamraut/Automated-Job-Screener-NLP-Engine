import streamlit as st
from typing import Dict, Any, List, Callable, Optional, Union
import re

def text_search_filter(items: List[Dict[str, Any]], 
                      search_term: str, 
                      search_fields: List[str] = ['original_text', 'processed.clean_text']):
    """
    Filter a list of items by search term
    
    Args:
        items: List of items to filter
        search_term: Search term to filter by
        search_fields: Fields to search in (supports dot notation for nested fields)
        
    Returns:
        List of filtered items
    """
    if not search_term:
        return items
    
    search_term = search_term.lower()
    filtered_items = []
    
    for item in items:
        for field in search_fields:
            # Handle nested fields with dot notation
            value = item
            for part in field.split('.'):
                if isinstance(value, dict) and part in value:
                    value = value[part]
                else:
                    value = None
                    break
            
            if value and isinstance(value, str) and search_term in value.lower():
                filtered_items.append(item)
                break
    
    return filtered_items

def skill_filter(items: List[Dict[str, Any]], required_skills: List[str]):
    """
    Filter items by required skills
    
    Args:
        items: List of items to filter
        required_skills: List of skills to require
        
    Returns:
        List of filtered items
    """
    if not required_skills:
        return items
    
    filtered_items = []
    required_skills_lower = [skill.lower() for skill in required_skills]
    
    for item in items:
        # Get skills from the item
        item_skills = []
        if 'processed' in item and 'entities' in item['processed'] and 'SKILL' in item['processed']['entities']:
            item_skills = [skill.lower() for skill in item['processed']['entities']['SKILL']]
        
        # Check if any required skills are present
        if any(skill in item_skills for skill in required_skills_lower):
            filtered_items.append(item)
    
    return filtered_items

def render_search_filters(on_change: Optional[Callable] = None) -> Dict[str, Any]:
    """
    Render search and filter controls
    
    Args:
        on_change: Optional callback when filters change
        
    Returns:
        Dictionary of active filters
    """
    filters = {}
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        search_term = st.text_input("Search", placeholder="Enter search terms")
        filters["search_term"] = search_term
    
    with col2:
        sort_by = st.selectbox("Sort by", ["Relevance", "Date", "Name"])
        filters["sort_by"] = sort_by
    
    # Expandable advanced filters
    with st.expander("Advanced Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        with col1:
            min_score = st.slider("Min. Score", 0.0, 1.0, 0.5, step=0.05)
            filters["min_score"] = min_score
            
            date_range = st.date_input("Date Range", [])
            filters["date_range"] = date_range
        
        with col2:
            skill_input = st.text_input("Required Skills (comma separated)")
            if skill_input:
                skills = [s.strip() for s in skill_input.split(",")]
                filters["required_skills"] = skills
            
            include_content = st.checkbox("Include full content in search", value=True)
            filters["include_content"] = include_content
    
    # Call the callback if provided
    if on_change:
        if st.button("Apply Filters"):
            on_change(filters)
    
    return filters