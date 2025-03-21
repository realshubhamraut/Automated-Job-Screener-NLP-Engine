import os
import json
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import re
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.config import (
    FINETUNED_SKILL_EXTRACTOR,
    FINETUNING_EPOCHS,
    FINETUNING_BATCH_SIZE,
    FINETUNING_LEARNING_RATE,
    RESUME_PROCESSED_DIR,
    JOB_DESC_PROCESSED_DIR
)

logger = get_logger(__name__)

class SkillExtractorTrainer:
    """Class for fine-tuning a skill extraction model"""
    
    def __init__(self, base_model: str = "distilbert-base-uncased"):
        """
        Initialize the skill extractor trainer
        
        Args:
            base_model: Pre-trained model to use as starting point
        """
        self.base_model = base_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing skill extractor trainer with model {base_model} on device {self.device}")
        
        # Define label mapping for NER
        self.id2label = {
            0: "O",        # Outside of any skill
            1: "B-SKILL",  # Beginning of a skill
            2: "I-SKILL"   # Inside of a skill
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    def prepare_training_data(self, labeled_data: List[Dict[str, Any]] = None) -> Dataset:
        """
        Prepare training data from labeled examples
        
        Args:
            labeled_data: List of {"text": str, "entities": List[Dict]} dictionaries
            
        Returns:
            HuggingFace Dataset for token classification
        """
        if labeled_data is None:
            # If no labeled data provided, generate it from existing data
            labeled_data = self.generate_labeled_data()
        
        # Process the data for token classification
        tokenized_inputs = []
        
        for example in labeled_data:
            text = example["text"]
            entities = example["entities"]
            
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            token_inputs = self.tokenizer(text, truncation=True, is_split_into_words=False)
            
            # Create labels aligned with tokens
            labels = ["O"] * len(tokens)
            
            # Mark skill entities in labels
            for entity in entities:
                if entity["label"] == "SKILL":
                    start = entity["start"]
                    end = entity["end"]
                    
                    # Find token indices that correspond to this entity
                    entity_tokens = self.tokenizer.tokenize(text[start:end])
                    
                    # Find where these tokens start in the full token list
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        if tokens[i:i+len(entity_tokens)] == entity_tokens:
                            # Mark as B-SKILL for first token, I-SKILL for rest
                            labels[i] = "B-SKILL"
                            for j in range(1, len(entity_tokens)):
                                if i + j < len(labels):
                                    labels[i + j] = "I-SKILL"
            
            # Convert string labels to IDs
            label_ids = [self.label2id.get(label, 0) for label in labels]
            
            # Add to tokenized inputs
            tokenized_input = {
                "input_ids": token_inputs["input_ids"],
                "attention_mask": token_inputs["attention_mask"],
                "labels": label_ids
            }
            
            tokenized_inputs.append(tokenized_input)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict({
            "input_ids": [x["input_ids"] for x in tokenized_inputs],
            "attention_mask": [x["attention_mask"] for x in tokenized_inputs],
            "labels": [x["labels"] for x in tokenized_inputs]
        })
        
        logger.info(f"Prepared dataset with {len(dataset)} examples")
        return dataset
    
    def generate_labeled_data(self) -> List[Dict[str, Any]]:
        """
        Generate labeled data from existing documents using rule-based approach
        
        Returns:
            List of {"text": str, "entities": List[Dict]} dictionaries
        """
        labeled_data = []
        
        try:
            # Define a preliminary list of technical skills
            from src.matching_engine.hybrid_search import HybridSearchEngine
            tech_terms = HybridSearchEngine.TECH_DOMAINS
            
            # Load resumes and job descriptions
            documents = []
            
            # Load resumes
            for filepath in Path(RESUME_PROCESSED_DIR).glob("*.json"):
                try:
                    with open(filepath, 'r') as f):
                        resume_data = json.load(f)
                    if 'processed' in resume_data and 'clean_text' in resume_data['processed']:
                        documents.append({
                            "text": resume_data['processed']['clean_text'],
                            "type": "resume"
                        })
                except Exception as e:
                    logger.error(f"Error loading resume {filepath}: {e}")
            
            # Load job descriptions
            for filepath in Path(JOB_DESC_PROCESSED_DIR).glob("*.json"):
                try:
                    with open(filepath, 'r') as f):
                        job_data = json.load(f)
                    if 'processed' in job_data and 'clean_text' in job_data['processed']:
                        documents.append({
                            "text": job_data['processed']['clean_text'],
                            "type": "job"
                        })
                except Exception as e:
                    logger.error(f"Error loading job description {filepath}: {e}")
            
            # Process each document to identify skills
            for doc in documents:
                text = doc["text"]
                entities = []
                
                # Find technical terms in the text
                for term in tech_terms:
                    term_pattern = r'\b' + re.escape(term) + r'\b'
                    for match in re.finditer(term_pattern, text, re.IGNORECASE):
                        entities.append({
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "label": "SKILL"
                        })
                
                # Add to labeled data
                if entities:
                    labeled_data.append({
                        "text": text,
                        "entities": entities
                    })
            
            logger.info(f"Generated {len(labeled_data)} labeled examples")
            
        except Exception as e:
            logger.error(f"Error generating labeled data: {e}")
        
        return labeled_data
    
    def fine_tune(self, dataset: Dataset,
                 epochs: int = FINETUNING_EPOCHS,
                 batch_size: int = FINETUNING_BATCH_SIZE,
                 learning_rate: float = FINETUNING_LEARNING_RATE,
                 output_path: str = FINETUNED_SKILL_EXTRACTOR):
        """
        Fine-tune the skill extraction model
        
        Args:
            dataset: Dataset for token classification
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            output_path: Path to save the fine-tuned model
        """
        if dataset is None or len(dataset) == 0:
            logger.error("No training data provided")
            return False
        
        try:
            logger.info(f"Starting fine-tuning with {len(dataset)} examples for {epochs} epochs")
            
            # Split dataset into train and eval
            train_test = dataset.train_test_split(test_size=0.1)
            train_dataset = train_test["train"]
            eval_dataset = train_test["test"]
            
            # Initialize model
            model = AutoModelForTokenClassification.from_pretrained(
                self.base_model, 
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # Set up data collator
            data_collator = DataCollatorForTokenClassification(self.tokenizer)
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=output_path,
# filepath: /Users/proxim/Desktop/Automated-First-Screener-and-AI-Job-Matching-NLP-WebEngine/src/training/skill_extractor_trainer.py

import os
import json
import torch
import logging
import numpy as np
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from transformers import (
    AutoModelForTokenClassification, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset
import re
from sklearn.model_selection import train_test_split

from src.utils.logger import get_logger
from src.config import (
    FINETUNED_SKILL_EXTRACTOR,
    FINETUNING_EPOCHS,
    FINETUNING_BATCH_SIZE,
    FINETUNING_LEARNING_RATE,
    RESUME_PROCESSED_DIR,
    JOB_DESC_PROCESSED_DIR
)

logger = get_logger(__name__)

class SkillExtractorTrainer:
    """Class for fine-tuning a skill extraction model"""
    
    def __init__(self, base_model: str = "distilbert-base-uncased"):
        """
        Initialize the skill extractor trainer
        
        Args:
            base_model: Pre-trained model to use as starting point
        """
        self.base_model = base_model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing skill extractor trainer with model {base_model} on device {self.device}")
        
        # Define label mapping for NER
        self.id2label = {
            0: "O",        # Outside of any skill
            1: "B-SKILL",  # Beginning of a skill
            2: "I-SKILL"   # Inside of a skill
        }
        self.label2id = {v: k for k, v in self.id2label.items()}
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    def prepare_training_data(self, labeled_data: List[Dict[str, Any]] = None) -> Dataset:
        """
        Prepare training data from labeled examples
        
        Args:
            labeled_data: List of {"text": str, "entities": List[Dict]} dictionaries
            
        Returns:
            HuggingFace Dataset for token classification
        """
        if labeled_data is None:
            # If no labeled data provided, generate it from existing data
            labeled_data = self.generate_labeled_data()
        
        # Process the data for token classification
        tokenized_inputs = []
        
        for example in labeled_data:
            text = example["text"]
            entities = example["entities"]
            
            # Tokenize the text
            tokens = self.tokenizer.tokenize(text)
            token_inputs = self.tokenizer(text, truncation=True, is_split_into_words=False)
            
            # Create labels aligned with tokens
            labels = ["O"] * len(tokens)
            
            # Mark skill entities in labels
            for entity in entities:
                if entity["label"] == "SKILL":
                    start = entity["start"]
                    end = entity["end"]
                    
                    # Find token indices that correspond to this entity
                    entity_tokens = self.tokenizer.tokenize(text[start:end])
                    
                    # Find where these tokens start in the full token list
                    for i in range(len(tokens) - len(entity_tokens) + 1):
                        if tokens[i:i+len(entity_tokens)] == entity_tokens:
                            # Mark as B-SKILL for first token, I-SKILL for rest
                            labels[i] = "B-SKILL"
                            for j in range(1, len(entity_tokens)):
                                if i + j < len(labels):
                                    labels[i + j] = "I-SKILL"
            
            # Convert string labels to IDs
            label_ids = [self.label2id.get(label, 0) for label in labels]
            
            # Add to tokenized inputs
            tokenized_input = {
                "input_ids": token_inputs["input_ids"],
                "attention_mask": token_inputs["attention_mask"],
                "labels": label_ids
            }
            
            tokenized_inputs.append(tokenized_input)
        
        # Convert to HuggingFace Dataset
        dataset = Dataset.from_dict({
            "input_ids": [x["input_ids"] for x in tokenized_inputs],
            "attention_mask": [x["attention_mask"] for x in tokenized_inputs],
            "labels": [x["labels"] for x in tokenized_inputs]
        })
        
        logger.info(f"Prepared dataset with {len(dataset)} examples")
        return dataset
    
    def generate_labeled_data(self) -> List[Dict[str, Any]]:
        """
        Generate labeled data from existing documents using rule-based approach
        
        Returns:
            List of {"text": str, "entities": List[Dict]} dictionaries
        """
        labeled_data = []
        
        try:
            # Define a preliminary list of technical skills
            from src.matching_engine.hybrid_search import HybridSearchEngine
            tech_terms = HybridSearchEngine.TECH_DOMAINS
            
            # Load resumes and job descriptions
            documents = []
            
            # Load resumes
            for filepath in Path(RESUME_PROCESSED_DIR).glob("*.json"):
                try:
                    with open(filepath, 'r') as f):
                        resume_data = json.load(f)
                    if 'processed' in resume_data and 'clean_text' in resume_data['processed']:
                        documents.append({
                            "text": resume_data['processed']['clean_text'],
                            "type": "resume"
                        })
                except Exception as e:
                    logger.error(f"Error loading resume {filepath}: {e}")
            
            # Load job descriptions
            for filepath in Path(JOB_DESC_PROCESSED_DIR).glob("*.json"):
                try:
                    with open(filepath, 'r') as f):
                        job_data = json.load(f)
                    if 'processed' in job_data and 'clean_text' in job_data['processed']:
                        documents.append({
                            "text": job_data['processed']['clean_text'],
                            "type": "job"
                        })
                except Exception as e:
                    logger.error(f"Error loading job description {filepath}: {e}")
            
            # Process each document to identify skills
            for doc in documents:
                text = doc["text"]
                entities = []
                
                # Find technical terms in the text
                for term in tech_terms:
                    term_pattern = r'\b' + re.escape(term) + r'\b'
                    for match in re.finditer(term_pattern, text, re.IGNORECASE):
                        entities.append({
                            "start": match.start(),
                            "end": match.end(),
                            "text": match.group(),
                            "label": "SKILL"
                        })
                
                # Add to labeled data
                if entities:
                    labeled_data.append({
                        "text": text,
                        "entities": entities
                    })
            
            logger.info(f"Generated {len(labeled_data)} labeled examples")
            
        except Exception as e:
            logger.error(f"Error generating labeled data: {e}")
        
        return labeled_data
    
    def fine_tune(self, dataset: Dataset,
                 epochs: int = FINETUNING_EPOCHS,
                 batch_size: int = FINETUNING_BATCH_SIZE,
                 learning_rate: float = FINETUNING_LEARNING_RATE,
                 output_path: str = FINETUNED_SKILL_EXTRACTOR):
        """
        Fine-tune the skill extraction model
        
        Args:
            dataset: Dataset for token classification
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            output_path: Path to save the fine-tuned model
        """
        if dataset is None or len(dataset) == 0:
            logger.error("No training data provided")
            return False
        
        try:
            logger.info(f"Starting fine-tuning with {len(dataset)} examples for {epochs} epochs")
            
            # Split dataset into train and eval
            train_test = dataset.train_test_split(test_size=0.1)
            train_dataset = train_test["train"]
            eval_dataset = train_test["test"]
            
            # Initialize model
            model = AutoModelForTokenClassification.from_pretrained(
                self.base_model, 
                num_labels=len(self.id2label),
                id2label=self.id2label,
                label2id=self.label2id
            )
            
            # Set up data collator
            data_collator = DataCollatorForTokenClassification(self.tokenizer)
            
            # Define training arguments
            training_args = TrainingArguments(
                output_dir=output_path,