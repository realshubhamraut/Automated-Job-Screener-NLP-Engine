import re
import string
import unicodedata
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem import WordNetLemmatizer
import spacy
import logging

from src.utils.logger import get_logger

logger = get_logger(__name__)

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

class TextPreprocessor:
    """Handles text preprocessing and cleaning for NLP tasks"""
    
    def __init__(self, language='english'):
        """
        Initialize the preprocessor
        
        Args:
            language (str): Language for stopwords and processing
        """
        self.language = language
        self.stop_words = set(stopwords.words(language))
        self.lemmatizer = WordNetLemmatizer()
        
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except:
            logger.warning("SpaCy model not found. Downloading en_core_web_sm...")
            spacy.cli.download("en_core_web_sm")
            self.nlp = spacy.load('en_core_web_sm')
            
    def clean_text(self, text):
        """
        Clean and normalize text
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
            
        # Convert to lowercase
        text = text.lower()
        
        # Replace multiple newlines and tabs with single space
        text = re.sub(r'[\n\r\t]+', ' ', text)
        
        # Remove URLs
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove phone numbers
        text = re.sub(r'\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', '', text)
        
        # Remove special characters but keep useful punctuation
        text = re.sub(r'[^a-zA-Z0-9.,;:!?\'\s]', ' ', text)
        
        # Replace multiple spaces with single space
        text = re.sub(r'\s+', ' ', text)
        
        return text.strip()
    
    def tokenize_text(self, text):
        """
        Tokenize text into words
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of tokens
        """
        return word_tokenize(text)
    
    def remove_stopwords(self, tokens):
        """
        Remove stopwords from tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Tokens with stopwords removed
        """
        return [token for token in tokens if token.lower() not in self.stop_words]
    
    def lemmatize_tokens(self, tokens):
        """
        Lemmatize tokens
        
        Args:
            tokens (list): List of tokens
            
        Returns:
            list: Lemmatized tokens
        """
        return [self.lemmatizer.lemmatize(token) for token in tokens]
    
    def normalize_text(self, text):
        """
        Normalize text (clean, tokenize, remove stopwords, lemmatize)
        
        Args:
            text (str): Raw input text
            
        Returns:
            str: Normalized text
        """
        cleaned_text = self.clean_text(text)
        tokens = self.tokenize_text(cleaned_text)
        filtered_tokens = self.remove_stopwords(tokens)
        lemmatized_tokens = self.lemmatize_tokens(filtered_tokens)
        
        return ' '.join(lemmatized_tokens)
        
    def extract_sentences(self, text):
        """
        Extract sentences from text
        
        Args:
            text (str): Input text
            
        Returns:
            list: List of sentences
        """
        return sent_tokenize(text)
    
    def extract_entities(self, text):
        """
        Extract named entities using spaCy
        
        Args:
            text (str): Input text
            
        Returns:
            dict: Dictionary of entities grouped by type
        """
        doc = self.nlp(text)
        entities = {}
        
        for ent in doc.ents:
            if ent.label_ not in entities:
                entities[ent.label_] = []
            entities[ent.label_].append(ent.text)
            
        return entities
        
    def process_document(self, text, include_entities=False):
        """
        Process a document - clean, tokenize, extract entities
        
        Args:
            text (str): Raw document text
            include_entities (bool): Whether to extract entities
            
        Returns:
            dict: Processed document with various fields
        """
        cleaned_text = self.clean_text(text)
        normalized_text = self.normalize_text(text)
        sentences = self.extract_sentences(cleaned_text)
        
        result = {
            'original_text': text,
            'cleaned_text': cleaned_text,
            'normalized_text': normalized_text,
            'sentences': sentences,
            'token_count': len(self.tokenize_text(cleaned_text)),
            'sentence_count': len(sentences),
        }
        
        if include_entities:
            result['entities'] = self.extract_entities(cleaned_text)
            
        return result