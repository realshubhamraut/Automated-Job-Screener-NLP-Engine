import numpy as np
from typing import Dict, List, Tuple, Any
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import importlib
import sys
import nltk

# Ensure NLTK is properly loaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from nltk.corpus import stopwords

# Force module reload to ensure changes are applied
if "src.matching_engine.hybrid_search" in sys.modules:
    importlib.reload(sys.modules["src.matching_engine.hybrid_search"])

class HybridSearchEngine:
    """Hybrid search engine that combines semantic and keyword-based search"""
    
    # Extended list of tech-specific stopwords
    TECH_STOPWORDS = [
        # Generic terms
        "city", "state", "member", "information", "management", "network", "maintained",
        "experience", "years", "year", "month", "months", "using", "use", "used", "work", 
        "worked", "working", "project", "projects", "team", "teams", "company", "companies",
        "position", "positions", "role", "roles", "job", "jobs", "responsibility", 
        "responsibilities", "duty", "duties", "task", "tasks", "environment", "implemented",
        "virtualized", "organizational", "strategic", "initiative", "established", "redundant",
        "communication", "technological", "creative", "networking", "relationship", "business",
        "solution", "solutions", "enterprise", "enterprises", "corporate", "corporation",
        "knowledge", "skills", "ability", "abilities", "competency", "competencies", 
        "qualification", "qualifications", "certified", "certification", "certifications",
        "excellent", "outstanding", "proficient", "proficiency", "expert", "expertise",
        "background", "professional", "entry", "level", "junior", "senior", "lead", "principal",
        "staff", "professionalism"
    ]
    
    # List of technical domains and specific technologies
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
    
    # Very specific technical keywords that should always be included
    DEFINITE_TECH_TERMS = {
        "api", "rest", "soap", "json", "xml", "yaml", "http", "https", "oauth",
        "jwt", "saml", "mfa", "sso", "aws", "azure", "gcp", "git", "github", "gitlab",
        "bitbucket", "jira", "kubernetes", "docker", "jenkins", "terraform", "ansible",
        "ci/cd", "devops", "agile", "scrum", "kanban", "react", "angular", "vue",
        "node.js", "java", "python", "c#", "c++", "ruby", "go", "kotlin", "swift",
        "sql", "nosql", "mongodb", "postgresql", "mysql", "redis", "kafka", "elasticsearch",
        "hadoop", "spark", "tensorflow", "pytorch", "ai", "ml", "microservices", "serverless",
        "lambda", "s3", "ec2", "rds"
    }

    def __init__(self, hybrid_weight: float = 0.7):
        """
        Initialize the hybrid search engine
        
        Args:
            hybrid_weight: Weight between semantic (1.0) and keyword (0.0) matching
        """
        self.hybrid_weight = hybrid_weight
        
        # Initialize keyword extraction tools
        try:
            self.stopwords = set(stopwords.words('english')).union(self.TECH_STOPWORDS)
        except:
            # Fallback if NLTK not available
            self.stopwords = set(self.TECH_STOPWORDS)
        
        # Create vectorizers for different n-gram ranges
        self.unigram_vectorizer = CountVectorizer(
            stop_words=self.stopwords,
            lowercase=True,
            ngram_range=(1, 1)
        )
        
        self.bigram_vectorizer = CountVectorizer(
            lowercase=True,
            ngram_range=(2, 2)
        )
        
        self.trigram_vectorizer = CountVectorizer(
            lowercase=True,
            ngram_range=(3, 3)
        )

    def hybrid_match(self, 
                     resume_embedding: List[float], 
                     job_embedding: List[float],
                     resume_text: str,
                     job_text: str) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate hybrid similarity between resume and job description
        
        Args:
            resume_embedding: Embedding vector for resume
            job_embedding: Embedding vector for job description
            resume_text: Cleaned resume text
            job_text: Cleaned job description text
            
        Returns:
            score: Combined similarity score
            details: Details of match including semantic and keyword scores
        """
        # Calculate semantic similarity using embeddings
        semantic_similarity = self._calculate_semantic_similarity(resume_embedding, job_embedding)
        
        # Calculate keyword similarity
        keyword_similarity, common_keywords, missing_keywords = self._calculate_keyword_similarity(resume_text, job_text)
        
        # Calculate combined score
        combined_score = (
            self.hybrid_weight * semantic_similarity + 
            (1 - self.hybrid_weight) * keyword_similarity
        )
        
        # Return combined score and details
        return combined_score, {
            "semantic_similarity": semantic_similarity,
            "keyword_similarity": keyword_similarity,
            "common_keywords": common_keywords,
            "missing_keywords": missing_keywords
        }
    
    def _calculate_semantic_similarity(self, 
                                      resume_embedding: List[float], 
                                      job_embedding: List[float]) -> float:
        """Calculate cosine similarity between embeddings"""
        resume_np = np.array(resume_embedding).reshape(1, -1)
        job_np = np.array(job_embedding).reshape(1, -1)
        return float(cosine_similarity(resume_np, job_np)[0][0])
    
    def _calculate_keyword_similarity(self, 
                                     resume_text: str, 
                                     job_text: str) -> Tuple[float, List[str], List[str]]:
        """
        Calculate keyword-based similarity and extract common and missing keywords
        
        Returns:
            similarity: Keyword similarity score
            common_keywords: Keywords found in both resume and job
            missing_keywords: Important keywords in job but missing in resume
        """
        # Extract technical terms with strict filtering
        resume_technical_terms = self._extract_technical_terms(resume_text)
        job_technical_terms = self._extract_technical_terms(job_text)
        
        # Find common and missing technical terms with case-insensitive matching
        resume_terms_lower = set(term.lower() for term in resume_technical_terms)
        job_terms_lower = set(term.lower() for term in job_technical_terms)
        
        common_terms_lower = resume_terms_lower.intersection(job_terms_lower)
        missing_terms_lower = job_terms_lower.difference(resume_terms_lower)
        
        # Calculate similarity as ratio of common terms to required terms
        if len(job_terms_lower) > 0:
            similarity = len(common_terms_lower) / len(job_terms_lower)
        else:
            similarity = 0.0
        
        # Get the original casing of common terms (from the job description for consistency)
        common_keywords = [
            term for term in job_technical_terms 
            if term.lower() in common_terms_lower
        ]
        
        # Get the missing keywords with original casing
        missing_keywords = [
            term for term in job_technical_terms
            if term.lower() in missing_terms_lower
        ]
        
        # Filter out any remaining non-technical terms
        common_keywords = [term for term in common_keywords if self._is_definitely_technical(term)]
        missing_keywords = [term for term in missing_keywords if self._is_definitely_technical(term)]
        
        return similarity, common_keywords, missing_keywords
    
    def _extract_technical_terms(self, text: str) -> List[str]:
        """
        Extract technical terms from text, focusing strictly on IT/Software terms
        """
        # Clean and normalize text
        text = text.lower()
        
        # First, try direct matching with our known tech domains list
        tech_matches = []
        
        # Direct term matching - look for exact technical terms
        for term in self.TECH_DOMAINS:
            if re.search(r'\b' + re.escape(term) + r'\b', text):
                tech_matches.append(term)
                
        # For short tech terms like "C++" that might not have word boundaries
        for term in self.DEFINITE_TECH_TERMS:
            if term in text:
                tech_matches.append(term)
        
        # Add n-gram analysis if direct matching doesn't find enough terms
        if len(tech_matches) < 10:
            # Extract all possible n-grams
            unigrams = self._extract_ngrams(text, self.unigram_vectorizer)
            bigrams = self._extract_ngrams(text, self.bigram_vectorizer)
            trigrams = self._extract_ngrams(text, self.trigram_vectorizer)
            
            # Filter and add relevant n-grams
            for term in unigrams + bigrams + trigrams:
                if self._is_definitely_technical(term) and term not in tech_matches:
                    tech_matches.append(term)
        
        # Sort by length (prioritize longer, more specific terms)
        tech_matches.sort(key=lambda x: (-len(x), x))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_tech_terms = []
        for term in tech_matches:
            if term.lower() not in seen:
                seen.add(term.lower())
                unique_tech_terms.append(term)
        
        # Return top 15 unique technical terms
        return unique_tech_terms[:15]
    
    def _extract_ngrams(self, text: str, vectorizer: CountVectorizer) -> List[str]:
        """Extract n-grams from text using the given vectorizer"""
        try:
            # Transform text using vectorizer
            X = vectorizer.fit_transform([text])
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Get non-zero features (terms that appear in the text)
            nonzero_cols = X.nonzero()[1]
            
            # Return terms that appear in the text
            return [feature_names[i] for i in nonzero_cols]
        except:
            return []
    
    def _is_technical_term(self, term: str) -> bool:
        """Basic check if a term could be technical"""
        term = term.lower()
        
        # Skip very short terms unless they are in our definite tech terms list
        if len(term) < 3 and term not in self.DEFINITE_TECH_TERMS:
            return False
        
        # Direct match with tech domains
        if term in self.TECH_DOMAINS:
            return True
            
        # Direct match with definite tech terms
        if term in self.DEFINITE_TECH_TERMS:
            return True
        
        # Check if term contains any tech domain
        for tech_term in self.TECH_DOMAINS:
            if tech_term in term or term in tech_term:
                return True
                
        # Skip business/generic terms
        for stop_term in self.TECH_STOPWORDS:
            if stop_term in term or term in stop_term:
                return False
        
        return False
    
    def _is_definitely_technical(self, term: str) -> bool:
        """Strict check to ensure a term is definitely technical"""
        term_lower = term.lower()
        
        # Always accept terms from our definite list
        if term_lower in self.DEFINITE_TECH_TERMS:
            return True
            
        # Always accept exact matches from tech domains
        if term_lower in self.TECH_DOMAINS:
            return True
            
        # Accept terms containing definite tech terms (like "aws lambda" containing "aws")
        for def_term in self.DEFINITE_TECH_TERMS:
            if def_term in term_lower:
                return True
                
        # Check for programming languages with versions
        if re.match(r'^(python|java|php|ruby)\s*[0-9.]+$', term_lower):
            return True
            
        # Check for technical frameworks with versions
        if re.match(r'^(react|angular|vue|node\.js|spring|rails)\s*[0-9.]+$', term_lower):
            return True
            
        # Check for common tech abbreviations
        if re.match(r'^[A-Za-z0-9\#\+\.]+$', term) and len(term) <= 8:
            if term_lower in {'api', 'rest', 'soap', 'json', 'xml', 'css', 'html', 'aws', 'gcp', 'ci', 'cd', 'ui', 'ux'}:
                return True
                
        # If term contains a stopword, it's likely not technical
        for stop_term in self.TECH_STOPWORDS:
            if stop_term in term_lower and len(stop_term) > 3:
                return False
                
        # Additional checks for specific patterns
        has_tech_term = False
        for tech_term in self.TECH_DOMAINS:
            if tech_term in term_lower:
                has_tech_term = True
                break
                
        return has_tech_term