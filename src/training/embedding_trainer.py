import os
import torch
import logging
import numpy as np
import json
from typing import List, Tuple, Optional
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm

from src.utils.logger import get_logger
from src.config import (
    EMBEDDING_MODEL, 
    FINETUNED_EMBEDDING_MODEL, 
    FINETUNING_EPOCHS, 
    FINETUNING_BATCH_SIZE,
    FINETUNING_LEARNING_RATE,
    RESUME_PROCESSED_DIR,
    JOB_DESC_PROCESSED_DIR
)

logger = get_logger(__name__)

class EmbeddingTrainer:
    """Class for fine-tuning embedding models on resume-job pairs"""
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize the embedding trainer
        
        Args:
            model_name: Pre-trained model to use as starting point
        """
        self.model_name = model_name
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"Initializing embedding trainer with model {model_name} on device {self.device}")
        
        # Load base model
        self.model = SentenceTransformer(model_name)
        
    def prepare_training_data(self, scored_pairs: List[Tuple[str, str, float]] = None) -> List[InputExample]:
        """
        Prepare training data from scored resume-job pairs
        
        Args:
            scored_pairs: List of (resume_text, job_text, similarity_score) tuples
            
        Returns:
            List of InputExample objects
        """
        if scored_pairs is None:
            # If no scored pairs provided, try to generate them from matched results
            scored_pairs = self.generate_training_pairs()
        
        # Convert to InputExample format for sentence-transformers
        train_examples = [
            InputExample(texts=[resume_text, job_text], label=float(score))
            for resume_text, job_text, score in scored_pairs
        ]
        
        logger.info(f"Prepared {len(train_examples)} training examples")
        return train_examples
    
    def generate_training_pairs(self) -> List[Tuple[str, str, float]]:
        """
        Generate training pairs from existing matches in the processed directories
        
        Returns:
            List of (resume_text, job_text, similarity_score) tuples
        """
        pairs = []
        
        try:
            # Load all resumes
            resumes = {}
            for filepath in Path(RESUME_PROCESSED_DIR).glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        resume_data = json.load(f)
                    if 'id' in resume_data and 'processed' in resume_data and 'clean_text' in resume_data['processed']:
                        resumes[resume_data['id']] = resume_data['processed']['clean_text']
                except Exception as e:
                    logger.error(f"Error loading resume {filepath}: {e}")
            
            # Load all job descriptions
            jobs = {}
            for filepath in Path(JOB_DESC_PROCESSED_DIR).glob("*.json"):
                try:
                    with open(filepath, 'r') as f:
                        job_data = json.load(f)
                    if 'id' in job_data and 'processed' in job_data and 'clean_text' in job_data['processed']:
                        jobs[job_data['id']] = job_data['processed']['clean_text']
                except Exception as e:
                    logger.error(f"Error loading job description {filepath}: {e}")
            
            # Check if any match results exist
            matches_path = Path(os.path.dirname(RESUME_PROCESSED_DIR)) / "matches"
            if matches_path.exists():
                for filepath in matches_path.glob("*.json"):
                    try:
                        with open(filepath, 'r') as f:
                            match_data = json.load(f)
                        
                        # Extract match pairs
                        for match in match_data.get('results', []):
                            resume_id = match.get('resume_id')
                            job_id = match.get('job_id')
                            score = match.get('score')
                            
                            if resume_id in resumes and job_id in jobs and score is not None:
                                pairs.append((resumes[resume_id], jobs[job_id], score))
                    except Exception as e:
                        logger.error(f"Error loading match data {filepath}: {e}")
            
            logger.info(f"Generated {len(pairs)} training pairs from existing matches")
            
            # If we don't have enough pairs, create synthetic pairs
            if len(pairs) < 100:
                logger.info("Not enough training pairs found. Generating synthetic pairs...")
                # Create synthetic pairs by randomly matching resumes and jobs
                # with estimated scores based on token overlap
                import random
                import re
                from sklearn.feature_extraction.text import CountVectorizer
                
                vectorizer = CountVectorizer(stop_words='english')
                
                resume_ids = list(resumes.keys())
                job_ids = list(jobs.keys())
                
                # Generate synthetic pairs
                synthetic_pairs = []
                for _ in range(min(500, len(resume_ids) * len(job_ids))):
                    resume_id = random.choice(resume_ids)
                    job_id = random.choice(job_ids)
                    
                    resume_text = resumes[resume_id]
                    job_text = jobs[job_id]
                    
                    # Calculate token overlap as proxy for similarity
                    try:
                        # Fit on concatenated texts to get shared vocabulary
                        X = vectorizer.fit_transform([resume_text, job_text])
                        
                        # Calculate cosine similarity
                        from sklearn.metrics.pairwise import cosine_similarity
                        sim_score = cosine_similarity(X[0:1], X[1:2])[0][0]
                        
                        synthetic_pairs.append((resume_text, job_text, sim_score))
                    except Exception as e:
                        logger.error(f"Error calculating synthetic similarity: {e}")
                
                # Add synthetic pairs to real pairs
                pairs.extend(synthetic_pairs)
                logger.info(f"Added {len(synthetic_pairs)} synthetic pairs")
        
        except Exception as e:
            logger.error(f"Error generating training pairs: {e}")
        
        return pairs
    
    def fine_tune(self, train_examples: List[InputExample], 
                 epochs: int = FINETUNING_EPOCHS, 
                 batch_size: int = FINETUNING_BATCH_SIZE,
                 learning_rate: float = FINETUNING_LEARNING_RATE,
                 output_path: str = FINETUNED_EMBEDDING_MODEL):
        """
        Fine-tune the embedding model
        
        Args:
            train_examples: Training examples
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            output_path: Path to save the fine-tuned model
        """
        if not train_examples:
            logger.error("No training examples provided")
            return False
        
        try:
            logger.info(f"Starting fine-tuning with {len(train_examples)} examples for {epochs} epochs")
            
            # Create data loader
            train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
            
            # Set up the loss function
            train_loss = losses.CosineSimilarityLoss(self.model)
            
            # Configure evaluation
            warmup_steps = int(len(train_dataloader) * epochs * 0.1)
            
            # Train the model
            self.model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=epochs,
                warmup_steps=warmup_steps,
                optimizer_params={'lr': learning_rate},
                output_path=output_path,
                show_progress_bar=True
            )
            
            logger.info(f"Model fine-tuning completed and saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error during fine-tuning: {e}")
            return False
    
    def evaluate(self, test_examples: List[InputExample], model_path: Optional[str] = None):
        """
        Evaluate a fine-tuned model
        
        Args:
            test_examples: Test examples for evaluation
            model_path: Path to model for evaluation (if None, use current model)
        
        Returns:
            Dictionary with evaluation metrics
        """
        if not test_examples:
            logger.error("No test examples provided")
            return {}
        
        try:
            # Load model if path provided
            if model_path:
                model = SentenceTransformer(model_path)
            else:
                model = self.model
                
            # Evaluate
            logger.info(f"Evaluating model on {len(test_examples)} examples")
            
            # Calculate cosine similarities between pairs
            similarities = []
            true_scores = []
            
            for example in tqdm(test_examples, desc="Evaluating"):
                texts = example.texts
                true_score = example.label
                
                # Get embeddings
                embeddings = model.encode(texts, convert_to_tensor=True)
                
                # Calculate cosine similarity
                similarity = torch.nn.functional.cosine_similarity(
                    embeddings[0].unsqueeze(0), 
                    embeddings[1].unsqueeze(0)
                ).item()
                
                similarities.append(similarity)
                true_scores.append(true_score)
            
            # Calculate metrics
            from scipy.stats import pearsonr, spearmanr
            
            pearson_corr, _ = pearsonr(similarities, true_scores)
            spearman_corr, _ = spearmanr(similarities, true_scores)
            
            metrics = {
                "pearson_correlation": pearson_corr,
                "spearman_correlation": spearman_corr,
                "num_examples": len(test_examples)
            }
            
            logger.info(f"Evaluation results: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {}