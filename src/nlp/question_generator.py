# import re
# import random
# from typing import List, Dict, Any, Optional
# from collections import Counter
# import spacy
# import nltk
# from nltk.corpus import stopwords
# import logging
# import os

# from src.utils.logger import get_logger

# # Initialize logger
# logger = get_logger(__name__)

# # Initialize NLP tools
# try:
#     nltk.download('stopwords', quiet=True)
#     nltk.download('punkt', quiet=True)
#     nltk.download('wordnet', quiet=True)
#     from nltk.tokenize import word_tokenize
    
#     # Load spaCy model if available
#     try:
#         import en_core_web_sm
#         nlp = en_core_web_sm.load()
#         SPACY_AVAILABLE = True
#     except ImportError:
#         try:
#             import spacy
#             nlp = spacy.load("en_core_web_sm")
#             SPACY_AVAILABLE = True
#         except:
#             logger.warning("spaCy model 'en_core_web_sm' not available. Using basic NLP processing.")
#             SPACY_AVAILABLE = False
# except Exception as e:
#     logger.error(f"Error initializing NLP tools: {str(e)}")
#     SPACY_AVAILABLE = False

# class QuestionGenerator:
#     """
#     Generate interview questions based on resume and job description content
#     using rule-based techniques when AI APIs are not available.
#     """
    
#     def __init__(self):
#         """Initialize the question generator with templates and configuration"""
#         # Question templates by type and difficulty
#         self.templates = {
#             'technical': {
#                 'basic': [
#                     "Can you explain what {keyword} is?",
#                     "What experience do you have with {keyword}?",
#                     "How would you rate your proficiency with {keyword}?",
#                     "Have you worked with {keyword} in your previous roles?"
#                 ],
#                 'intermediate': [
#                     "Can you describe a project where you implemented {keyword}?",
#                     "What challenges have you faced when working with {keyword}?",
#                     "How do you stay updated with the latest developments in {keyword}?",
#                     "Explain how {keyword} works and where you've applied it"
#                 ],
#                 'advanced': [
#                     "Describe a complex problem you solved using {keyword}",
#                     "How would you optimize a system that uses {keyword}?",
#                     "What are the tradeoffs when implementing {keyword} versus alternative approaches?",
#                     "If you were to build a new solution using {keyword}, how would you approach it?"
#                 ],
#                 'expert': [
#                     "What innovations or improvements have you personally made when working with {keyword}?",
#                     "Explain how you would architect a system using {keyword} at enterprise scale",
#                     "How would you debug a complex issue in a system that uses {keyword}?",
#                     "What are the cutting-edge developments in {keyword} and how might they impact the industry?"
#                 ]
#             },
#             'experience': {
#                 'basic': [
#                     "Tell me about your experience with {keyword}",
#                     "How long have you worked with {keyword}?",
#                     "What roles have you used {keyword} in?",
#                     "What aspects of {keyword} are you most familiar with?"
#                 ],
#                 'intermediate': [
#                     "Describe a project where your experience with {keyword} made a significant impact",
#                     "How has your approach to using {keyword} evolved over your career?",
#                     "What lessons have you learned while working with {keyword}?",
#                     "How have you applied {keyword} in different contexts or industries?"
#                 ],
#                 'advanced': [
#                     "What's the most challenging aspect of {keyword} you've dealt with?",
#                     "Tell me about a time when your expertise in {keyword} helped solve a critical business problem",
#                     "How have you leveraged {keyword} to drive innovation?",
#                     "Describe how you've trained or mentored others in using {keyword}"
#                 ],
#                 'expert': [
#                     "How have you contributed to the evolution or improvement of {keyword} in your field?",
#                     "Describe how you've integrated {keyword} with other technologies to create novel solutions",
#                     "What thought leadership have you provided regarding {keyword}?",
#                     "How have you influenced organizational strategy through your expertise in {keyword}?"
#                 ]
#             },
#             'problem solving': {
#                 'basic': [
#                     "How would you approach a problem related to {keyword}?",
#                     "What steps would you take to implement a solution using {keyword}?",
#                     "How do you troubleshoot issues with {keyword}?",
#                     "What resources do you use when facing challenges with {keyword}?"
#                 ],
#                 'intermediate': [
#                     "Describe a situation where you had to solve a complex problem using {keyword}",
#                     "How do you evaluate different approaches when using {keyword} to solve a problem?",
#                     "What methodology do you follow when implementing solutions with {keyword}?",
#                     "How do you balance technical constraints and business needs when working with {keyword}?"
#                 ],
#                 'advanced': [
#                     "Tell me about a time when conventional approaches to {keyword} failed and how you innovated",
#                     "How have you optimized or improved processes related to {keyword}?",
#                     "What's your approach to solving ambiguous problems where {keyword} might be applicable?",
#                     "Describe how you've solved scalability or performance challenges related to {keyword}"
#                 ],
#                 'expert': [
#                     "How would you approach an unsolved problem in the domain of {keyword}?",
#                     "Describe how you would design a solution for a novel use case of {keyword}",
#                     "How do you evaluate risk when pioneering new applications of {keyword}?",
#                     "What framework would you create to solve a class of problems related to {keyword}?"
#                 ]
#             },
#             'role-specific': {
#                 'basic': [
#                     "Why are you interested in a role that involves {keyword}?",
#                     "How does {keyword} align with your career goals?",
#                     "What attracted you to working with {keyword} in this role?",
#                     "How do you see yourself contributing to our team in the area of {keyword}?"
#                 ],
#                 'intermediate': [
#                     "How would you apply your experience with {keyword} to this specific role?",
#                     "What unique perspective do you bring to {keyword} that would benefit our team?",
#                     "How would you prioritize tasks related to {keyword} in this position?",
#                     "How would you collaborate with team members on projects involving {keyword}?"
#                 ],
#                 'advanced': [
#                     "How would you improve our approach to {keyword} based on what you know about our company?",
#                     "What innovations could you bring to our use of {keyword}?",
#                     "How would you align {keyword} initiatives with our business objectives?",
#                     "What metrics would you establish to measure success in {keyword} initiatives?"
#                 ],
#                 'expert': [
#                     "How would you develop a strategic roadmap for {keyword} initiatives in our organization?",
#                     "How would you transform our approach to {keyword} to gain competitive advantage?",
#                     "What organizational changes would you recommend to better leverage {keyword}?",
#                     "How would you build and lead a center of excellence around {keyword}?"
#                 ]
#             },
#             'behavioral': {
#                 'basic': [
#                     "Tell me about a time when you worked with {keyword}",
#                     "How do you approach learning new aspects of {keyword}?",
#                     "How do you handle challenges related to {keyword}?",
#                     "How do you collaborate with others when working on {keyword}?"
#                 ],
#                 'intermediate': [
#                     "Describe a situation where you had to advocate for a particular approach to {keyword}",
#                     "Tell me about a time when you received critical feedback about your work with {keyword}",
#                     "How have you handled disagreements about the use of {keyword}?",
#                     "Describe a time when you had to make a difficult decision related to {keyword}"
#                 ],
#                 'advanced': [
#                     "Tell me about a time when you led a team working on {keyword}",
#                     "Describe a situation where you had to manage conflicting priorities involving {keyword}",
#                     "How have you influenced others to adopt best practices related to {keyword}?",
#                     "Describe a time when you had to adapt quickly to changes involving {keyword}"
#                 ],
#                 'expert': [
#                     "Tell me about a time when you drove organizational change related to {keyword}",
#                     "Describe how you've mentored leaders in your organization on matters related to {keyword}",
#                     "How have you handled resistance to your strategic vision for {keyword}?",
#                     "Tell me about a time when you had to make an unpopular decision about {keyword} that proved to be correct"
#                 ]
#             },
#             'culture fit': {
#                 'basic': [
#                     "How do you stay current with developments in {keyword}?",
#                     "What do you enjoy most about working with {keyword}?",
#                     "How do you approach collaboration on projects involving {keyword}?",
#                     "What values do you think are important when working with {keyword}?"
#                 ],
#                 'intermediate': [
#                     "How do you balance quality and speed when working with {keyword}?",
#                     "How do you approach knowledge sharing about {keyword} with team members?",
#                     "What kind of team environment helps you do your best work with {keyword}?",
#                     "How do you handle situations when team members have different levels of expertise in {keyword}?"
#                 ],
#                 'advanced': [
#                     "How have you promoted a positive culture while leading teams working on {keyword}?",
#                     "How do you approach mentoring junior team members on {keyword}?",
#                     "How do you balance innovation and stability when working with {keyword}?",
#                     "How have you fostered diversity of thought when approaching problems related to {keyword}?"
#                 ],
#                 'expert': [
#                     "How have you shaped organizational culture around the use of {keyword}?",
#                     "How do you balance business needs and team wellbeing when driving initiatives around {keyword}?",
#                     "How have you built inclusive leadership practices in teams focused on {keyword}?",
#                     "How do you approach succession planning for critical roles involving {keyword}?"
#                 ]
#             }
#         }
        
#         # Generic questions that don't require keywords
#         self.generic_questions = {
#             'technical': [
#                 "What technical skills are you most proud of?",
#                 "How do you approach learning new technologies?",
#                 "What development methodologies are you familiar with?",
#                 "How do you stay updated with industry trends?"
#             ],
#             'experience': [
#                 "What has been your most challenging project so far?",
#                 "Describe your ideal work environment",
#                 "What achievements are you most proud of?",
#                 "How has your past experience prepared you for this role?"
#             ],
#             'problem solving': [
#                 "Describe your approach to solving complex problems",
#                 "How do you make decisions when you don't have all the information?",
#                 "Tell me about a time when you had to think outside the box",
#                 "How do you prioritize when handling multiple tasks?"
#             ],
#             'role-specific': [
#                 "Why are you interested in this position?",
#                 "How do you see yourself contributing to our team?",
#                 "What aspects of this role excite you the most?",
#                 "How does this position align with your career goals?"
#             ],
#             'behavioral': [
#                 "Tell me about a time when you had to deal with a difficult teammate",
#                 "How do you handle feedback?",
#                 "Describe a situation where you demonstrated leadership",
#                 "How do you handle tight deadlines?"
#             ],
#             'culture fit': [
#                 "What's your ideal company culture?",
#                 "How do you prefer to receive feedback?",
#                 "What values are most important to you in a workplace?",
#                 "How do you approach work-life balance?"
#             ]
#         }
        
#         # Stop words for keyword extraction
#         self.stop_words = set(stopwords.words('english') if 'stopwords' in nltk.data.path else [])
#         self.stop_words.update([
#             'experience', 'skill', 'year', 'work', 'job', 'role', 'position',
#             'candidate', 'ability', 'team', 'project', 'using', 'used', 'use',
#             'required', 'requirement', 'require', 'qualifications', 'qualified',
#             'include', 'including', 'develop', 'developing', 'development',
#             'responsible', 'responsibility', 'prefer', 'preferred', 'plus'
#         ])
        
#     def generate_questions(
#         self, 
#         resume_text: str, 
#         job_text: str, 
#         num_questions: int = 8, 
#         question_types: List[str] = None,
#         difficulty: str = 'intermediate'
#     ) -> List[Dict[str, str]]:
#         """
#         Generate interview questions based on resume and job description
        
#         Args:
#             resume_text: The text content of the resume
#             job_text: The text content of the job description
#             num_questions: Number of questions to generate
#             question_types: Types of questions to generate (default: all types)
#             difficulty: Difficulty level (basic, intermediate, advanced, expert)
            
#         Returns:
#             List of question dictionaries
#         """
#         try:
#             # Normalize inputs
#             if not question_types:
#                 question_types = list(self.templates.keys())
#             else:
#                 # Normalize question types to match our template keys
#                 question_types = [qt.lower() for qt in question_types]
            
#             # Normalize difficulty
#             difficulty = difficulty.lower()
#             if difficulty not in ['basic', 'intermediate', 'advanced', 'expert']:
#                 difficulty = 'intermediate'
                
#             # Extract keywords
#             keywords = self._extract_keywords(resume_text, job_text)
            
#             # Generate questions
#             questions = []
            
#             # Calculate questions per type
#             questions_per_type = num_questions // len(question_types)
#             remainder = num_questions % len(question_types)
            
#             # Generate questions for each type
#             for i, q_type in enumerate(question_types):
#                 # Determine how many questions for this type
#                 type_count = questions_per_type
#                 if i < remainder:
#                     type_count += 1
                    
#                 # Skip if count is 0
#                 if type_count <= 0:
#                     continue
                    
#                 # Generate questions of this type
#                 type_questions = self._generate_questions_by_type(
#                     q_type, difficulty, keywords, type_count
#                 )
#                 questions.extend(type_questions)
            
#             # If we couldn't generate enough questions, add generic ones
#             if len(questions) < num_questions:
#                 needed = num_questions - len(questions)
#                 generic = self._generate_generic_questions(question_types, needed)
#                 questions.extend(generic)
                
#             # Shuffle and limit to requested number
#             random.shuffle(questions)
#             return questions[:num_questions]
            
#         except Exception as e:
#             logger.error(f"Error generating questions: {str(e)}")
#             # Return basic generic questions as fallback
#             return self._generate_generic_questions(['technical', 'experience'], num_questions)
    
#     def _extract_keywords(self, resume_text: str, job_text: str, top_n: int = 15) -> List[str]:
#         """
#         Extract relevant keywords from resume and job description
        
#         Args:
#             resume_text: Resume text
#             job_text: Job description text
#             top_n: Number of top keywords to return
            
#         Returns:
#             List of keywords
#         """
#         try:
#             # Use spaCy for better NER and keyword extraction if available
#             if SPACY_AVAILABLE:
#                 return self._extract_keywords_spacy(resume_text, job_text, top_n)
#             else:
#                 return self._extract_keywords_basic(resume_text, job_text, top_n)
#         except Exception as e:
#             logger.error(f"Error extracting keywords: {str(e)}")
#             # Return empty list if extraction fails
#             return []
    
#     def _extract_keywords_spacy(self, resume_text: str, job_text: str, top_n: int) -> List[str]:
#         """Extract keywords using spaCy"""
#         # Process texts
#         resume_doc = nlp(resume_text)
#         job_doc = nlp(job_text)
        
#         # Extract named entities, noun phrases, and technical terms
#         keywords = []
        
#         # From resume
#         for ent in resume_doc.ents:
#             if ent.label_ in ["SKILL", "ORG", "PRODUCT", "WORK_OF_ART", "GPE", "EVENT"]:
#                 keywords.append(ent.text.lower())
                
#         # From job description
#         for ent in job_doc.ents:
#             if ent.label_ in ["SKILL", "ORG", "PRODUCT", "WORK_OF_ART", "GPE", "EVENT"]:
#                 keywords.append(ent.text.lower())
        
#         # Extract noun phrases as potential skills/technologies
#         for np in resume_doc.noun_chunks:
#             if len(np.text.split()) <= 3:  # Limit to shorter phrases
#                 keywords.append(np.text.lower())
                
#         for np in job_doc.noun_chunks:
#             if len(np.text.split()) <= 3:  # Limit to shorter phrases
#                 keywords.append(np.text.lower())
        
#         # Count frequencies and get top keywords
#         keyword_counter = Counter(keywords)
        
#         # Filter common words and short terms
#         filtered_keywords = [k for k, _ in keyword_counter.most_common(top_n*2) 
#                           if k not in self.stop_words and len(k) > 2]
                          
#         return filtered_keywords[:top_n]
    
#     def _extract_keywords_basic(self, resume_text: str, job_text: str, top_n: int) -> List[str]:
#         """Extract keywords using basic NLP techniques"""
#         # Combine texts with more weight to job description
#         combined_text = resume_text + " " + job_text + " " + job_text
        
#         # Tokenize and lowercase
#         tokens = word_tokenize(combined_text.lower()) if 'punkt' in nltk.data.path else combined_text.lower().split()
        
#         # Remove stopwords and punctuation
#         tokens = [token for token in tokens if token.isalnum() and token not in self.stop_words and len(token) > 2]
        
#         # Count token frequencies
#         counter = Counter(tokens)
        
#         # Get most common words
#         common_words = [word for word, _ in counter.most_common(top_n)]
        
#         # Extract bigrams and trigrams (simple approach)
#         words = combined_text.lower().split()
#         bigrams = [' '.join(words[i:i+2]) for i in range(len(words)-1)]
#         trigrams = [' '.join(words[i:i+3]) for i in range(len(words)-2)]
        
#         # Filter n-grams
#         filtered_bigrams = [bg for bg in bigrams 
#                          if not all(w in self.stop_words for w in bg.split()) 
#                          and len(bg) > 5]
#         filtered_trigrams = [tg for tg in trigrams 
#                           if not all(w in self.stop_words for w in tg.split())
#                           and len(tg) > 7]
        
#         # Count n-grams
#         bigram_counter = Counter(filtered_bigrams)
#         trigram_counter = Counter(filtered_trigrams)
        
#         # Get top n-grams
#         top_bigrams = [b for b, _ in bigram_counter.most_common(top_n//3)]
#         top_trigrams = [t for t, _ in trigram_counter.most_common(top_n//3)]
        
#         # Combine and return unique keywords
#         all_keywords = common_words + top_bigrams + top_trigrams
#         unique_keywords = list(dict.fromkeys(all_keywords))  # Remove duplicates while preserving order
        
#         return unique_keywords[:top_n]
    
#     def _generate_questions_by_type(
#         self, 
#         question_type: str, 
#         difficulty: str, 
#         keywords: List[str], 
#         count: int
#     ) -> List[Dict[str, str]]:
#         """Generate questions of a specific type and difficulty"""
#         questions = []
        
#         # Check if we have templates for this type
#         if question_type not in self.templates:
#             return []
            
#         # Get templates for this type and difficulty
#         templates = self.templates[question_type].get(difficulty, self.templates[question_type]['intermediate'])
        
#         # Generate questions using keywords
#         keyword_idx = 0
#         for _ in range(count):
#             if keyword_idx >= len(keywords) or not keywords:
#                 # Use generic questions if we run out of keywords
#                 if len(questions) < count and question_type in self.generic_questions:
#                     generic_q = random.choice(self.generic_questions[question_type])
#                     questions.append({
#                         'question': generic_q,
#                         'type': question_type.capitalize(),
#                         'difficulty': difficulty.capitalize(),
#                         'purpose': f"To assess candidate's {question_type} skills",
#                         'good_answer_criteria': f"Clear, specific response demonstrating {question_type} capabilities"
#                     })
#                 continue
                
#             # Get a random template and keyword
#             template = random.choice(templates)
#             keyword = keywords[keyword_idx]
#             keyword_idx += 1
            
#             # Format the template with the keyword
#             question_text = template.format(keyword=keyword)
            
#             # Create question object
#             question = {
#                 'question': question_text,
#                 'type': question_type.capitalize(),
#                 'difficulty': difficulty.capitalize(),
#                 'purpose': f"To assess candidate's knowledge and experience with {keyword}",
#                 'good_answer_criteria': f"Demonstrates concrete experience with {keyword} through specific examples"
#             }
            
#             questions.append(question)
        
#         return questions
    
#     def _generate_generic_questions(self, question_types: List[str], count: int) -> List[Dict[str, str]]:
#         """Generate generic questions when specific keywords aren't available"""
#         questions = []
        
#         # How many questions per type
#         questions_per_type = max(1, count // len(question_types))
        
#         for q_type in question_types:
#             # Skip types we don't have templates for
#             if q_type not in self.generic_questions:
#                 continue
                
#             # Get generic questions for this type
#             type_questions = self.generic_questions[q_type]
            
#             # Select random questions
#             selected = random.sample(
#                 type_questions, 
#                 min(questions_per_type, len(type_questions))
#             )
            
#             # Create question objects
#             for q_text in selected:
#                 question = {
#                     'question': q_text,
#                     'type': q_type.capitalize(),
#                     'difficulty': 'Intermediate',  # Default to intermediate for generic
#                     'purpose': f"To assess candidate's general {q_type} abilities",
#                     'good_answer_criteria': f"Clear, specific answer with relevant examples"
#                 }
                
#                 questions.append(question)
                
#                 # Break if we have enough questions
#                 if len(questions) >= count:
#                     break
                    
#             # Break if we have enough questions
#             if len(questions) >= count:
#                 break
        
#         # If we still don't have enough, add some general fallbacks
#         fallbacks = [
#             {
#                 'question': "Tell me about yourself and your background",
#                 'type': "General",
#                 'difficulty': "Basic",
#                 'purpose': "To understand candidate's overall experience and background",
#                 'good_answer_criteria': "Concise overview of relevant experience and skills"
#             },
#             {
#                 'question': "Why are you interested in this position?",
#                 'type': "Motivation",
#                 'difficulty': "Basic",
#                 'purpose': "To assess candidate's interest and motivation",
#                 'good_answer_criteria': "Shows genuine interest and knowledge about the role and company"
#             },
#             {
#                 'question': "What are your strengths and weaknesses?",
#                 'type': "Self-awareness",
#                 'difficulty': "Basic",
#                 'purpose': "To assess candidate's self-awareness and honesty",
#                 'good_answer_criteria': "Honest assessment with examples and growth mindset"
#             },
#             {
#                 'question': "Where do you see yourself in five years?",
#                 'type': "Career Goals",
#                 'difficulty': "Basic",
#                 'purpose': "To understand candidate's career aspirations",
#                 'good_answer_criteria': "Realistic goals that align with the position and company"
#             }
#         ]
        
#         # Add fallbacks if needed
#         while len(questions) < count:
#             questions.append(fallbacks[len(questions) % len(fallbacks)])
        
#         return questions[:count]