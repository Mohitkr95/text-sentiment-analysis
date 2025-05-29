import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from typing import Tuple, List
import pickle

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import MAX_FEATURES, NGRAM_RANGE, MIN_DF, MAX_DF
from src.utils.logger import setup_logger

class FeatureExtractor:
    """Advanced feature extraction for sentiment analysis."""
    
    def __init__(self, use_tfidf: bool = True, use_additional_features: bool = True):
        self.logger = setup_logger(self.__class__.__name__)
        self.use_tfidf = use_tfidf
        self.use_additional_features = use_additional_features
        
        # Initialize vectorizers
        if use_tfidf:
            self.vectorizer = TfidfVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=NGRAM_RANGE,
                min_df=MIN_DF,
                max_df=MAX_DF,
                lowercase=False,  # Already lowercased in preprocessing
                stop_words=None,  # Already removed in preprocessing
                token_pattern=r'\b\w+\b'
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=MAX_FEATURES,
                ngram_range=NGRAM_RANGE,
                min_df=MIN_DF,
                max_df=MAX_DF,
                lowercase=False,
                stop_words=None,
                token_pattern=r'\b\w+\b'
            )
        
        self.feature_selector = None
        self.is_fitted = False
        
    def extract_additional_features(self, texts: List[str]) -> np.ndarray:
        """Extract additional text features beyond bag-of-words."""
        features = []
        
        for text in texts:
            text_features = []
            
            # Length features
            text_features.append(len(text))  # Character count
            text_features.append(len(text.split()))  # Word count
            
            # Punctuation features
            text_features.append(text.count('!'))  # Exclamation marks
            text_features.append(text.count('?'))  # Question marks
            text_features.append(text.count('.'))  # Periods
            
            # Sentiment indicators
            text_features.append(text.count('not'))  # Negation
            text_features.append(text.count('very'))  # Intensifier
            text_features.append(text.count('really'))  # Intensifier
            
            # Capital letters (original text)
            original_text = text  # Assuming this is passed separately
            text_features.append(sum(1 for c in original_text if c.isupper()))
            
            # Average word length
            words = text.split()
            avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
            text_features.append(avg_word_length)
            
            features.append(text_features)
        
        return np.array(features)
    
    def fit_transform(self, texts: List[str], labels: np.ndarray = None) -> np.ndarray:
        """
        Fit the feature extractor and transform texts to features.
        
        Args:
            texts: List of preprocessed texts
            labels: Optional labels for feature selection
            
        Returns:
            Feature matrix
        """
        self.logger.info("Fitting feature extractor and transforming texts...")
        
        # Extract bag-of-words/TF-IDF features
        bow_features = self.vectorizer.fit_transform(texts)
        
        # Get feature names for logging
        feature_names = self.vectorizer.get_feature_names_out()
        self.logger.info(f"Extracted {len(feature_names)} {('TF-IDF' if self.use_tfidf else 'bag-of-words')} features")
        
        # Feature selection if labels provided
        if labels is not None:
            self.logger.info("Performing feature selection...")
            self.feature_selector = SelectKBest(chi2, k=min(3000, bow_features.shape[1]))
            bow_features = self.feature_selector.fit_transform(bow_features, labels)
            self.logger.info(f"Selected {bow_features.shape[1]} best features")
        
        # Extract additional features if enabled
        if self.use_additional_features:
            additional_features = self.extract_additional_features(texts)
            self.logger.info(f"Extracted {additional_features.shape[1]} additional features")
            
            # Combine features
            combined_features = np.hstack([bow_features.toarray(), additional_features])
        else:
            combined_features = bow_features.toarray()
        
        self.is_fitted = True
        self.logger.info(f"Final feature matrix shape: {combined_features.shape}")
        
        return combined_features
    
    def transform(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to features using fitted extractor.
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            Feature matrix
        """
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before transform")
        
        # Extract bag-of-words/TF-IDF features
        bow_features = self.vectorizer.transform(texts)
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            bow_features = self.feature_selector.transform(bow_features)
        
        # Extract additional features if enabled
        if self.use_additional_features:
            additional_features = self.extract_additional_features(texts)
            # Combine features
            combined_features = np.hstack([bow_features.toarray(), additional_features])
        else:
            combined_features = bow_features.toarray()
        
        return combined_features
    
    def get_feature_names(self) -> List[str]:
        """Get feature names."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before getting feature names")
        
        # Get bag-of-words feature names
        feature_names = list(self.vectorizer.get_feature_names_out())
        
        # Apply feature selection if fitted
        if self.feature_selector is not None:
            selected_indices = self.feature_selector.get_support(indices=True)
            feature_names = [feature_names[i] for i in selected_indices]
        
        # Add additional feature names if enabled
        if self.use_additional_features:
            additional_names = [
                'char_count', 'word_count', 'exclamation_count', 
                'question_count', 'period_count', 'not_count',
                'very_count', 'really_count', 'capital_count', 'avg_word_length'
            ]
            feature_names.extend(additional_names)
        
        return feature_names
    
    def save(self, filepath: str):
        """Save the fitted feature extractor."""
        if not self.is_fitted:
            raise ValueError("FeatureExtractor must be fitted before saving")
        
        save_data = {
            'vectorizer': self.vectorizer,
            'feature_selector': self.feature_selector,
            'use_tfidf': self.use_tfidf,
            'use_additional_features': self.use_additional_features,
            'is_fitted': self.is_fitted
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_data, f)
        
        self.logger.info(f"Feature extractor saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'FeatureExtractor':
        """Load a fitted feature extractor."""
        with open(filepath, 'rb') as f:
            save_data = pickle.load(f)
        
        extractor = cls(
            use_tfidf=save_data['use_tfidf'],
            use_additional_features=save_data['use_additional_features']
        )
        
        extractor.vectorizer = save_data['vectorizer']
        extractor.feature_selector = save_data['feature_selector']
        extractor.is_fitted = save_data['is_fitted']
        
        return extractor 