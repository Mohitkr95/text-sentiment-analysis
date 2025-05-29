import re
import string
import sys
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from textblob import TextBlob
from tqdm import tqdm

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import MIN_WORD_LENGTH
from src.utils.logger import setup_logger

class TextProcessor:
    """Advanced text preprocessing for sentiment analysis."""
    
    def __init__(self):
        self.logger = setup_logger(self.__class__.__name__)
        self._download_nltk_data()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Add sentiment-specific stop words to keep
        sentiment_words = {'not', 'no', 'never', 'nothing', 'neither', 'nobody', 
                          'none', 'nowhere', 'without', 'barely', 'hardly', 
                          'scarcely', 'seldom', 'rarely'}
        self.stop_words = self.stop_words - sentiment_words
        
    def _download_nltk_data(self):
        """Download required NLTK data."""
        try:
            nltk.data.find('tokenizers/punkt')
            nltk.data.find('corpora/stopwords')
            nltk.data.find('corpora/wordnet')
            nltk.data.find('taggers/averaged_perceptron_tagger')
        except LookupError:
            self.logger.info("Downloading required NLTK data...")
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('averaged_perceptron_tagger', quiet=True)
            nltk.download('omw-1.4', quiet=True)
    
    def clean_text(self, text: str) -> str:
        """
        Clean and preprocess a single text.
        
        Args:
            text: Raw text string
            
        Returns:
            Cleaned text string
        """
        if pd.isna(text) or not isinstance(text, str):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Handle contractions
        text = self._expand_contractions(text)
        
        # Remove special characters but keep emoticons
        text = re.sub(r'[^a-zA-Z\s!?.:;]', ' ', text)
        
        # Handle multiple punctuation
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        text = re.sub(r'[.]{2,}', '.', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def _expand_contractions(self, text: str) -> str:
        """Expand common contractions."""
        contractions = {
            "n't": " not", "'re": " are", "'ve": " have", "'ll": " will",
            "'d": " would", "'m": " am", "can't": "cannot", "won't": "will not",
            "shouldn't": "should not", "wouldn't": "would not", "couldn't": "could not",
            "doesn't": "does not", "don't": "do not", "isn't": "is not",
            "aren't": "are not", "wasn't": "was not", "weren't": "were not",
            "hasn't": "has not", "haven't": "have not", "hadn't": "had not"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        return text
    
    def tokenize_and_lemmatize(self, text: str) -> List[str]:
        """
        Tokenize and lemmatize text.
        
        Args:
            text: Cleaned text string
            
        Returns:
            List of lemmatized tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and short words
        tokens = [token for token in tokens 
                 if token not in self.stop_words 
                 and len(token) >= MIN_WORD_LENGTH
                 and token not in string.punctuation]
        
        # Lemmatize
        lemmatized_tokens = []
        for token in tokens:
            try:
                lemmatized_token = self.lemmatizer.lemmatize(token, pos='v')  # verb
                if lemmatized_token == token:
                    lemmatized_token = self.lemmatizer.lemmatize(token, pos='n')  # noun
                lemmatized_tokens.append(lemmatized_token)
            except:
                lemmatized_tokens.append(token)
        
        return lemmatized_tokens
    
    def correct_spelling(self, text: str) -> str:
        """Basic spelling correction using TextBlob."""
        try:
            blob = TextBlob(text)
            corrected = str(blob.correct())
            return corrected
        except:
            return text
    
    def process_corpus(self, texts: List[str], correct_spelling: bool = False) -> List[str]:
        """
        Process a corpus of texts.
        
        Args:
            texts: List of raw text strings
            correct_spelling: Whether to apply spelling correction
            
        Returns:
            List of processed text strings
        """
        self.logger.info(f"Processing corpus of {len(texts)} texts...")
        
        processed_texts = []
        
        for text in tqdm(texts, desc="Processing texts"):
            # Clean text
            cleaned = self.clean_text(text)
            
            # Optional spelling correction (slow)
            if correct_spelling:
                cleaned = self.correct_spelling(cleaned)
            
            # Tokenize and lemmatize
            tokens = self.tokenize_and_lemmatize(cleaned)
            
            # Join tokens back to string
            processed_text = ' '.join(tokens)
            processed_texts.append(processed_text)
        
        self.logger.info("Text processing completed!")
        return processed_texts
    
    def get_text_statistics(self, texts: List[str]) -> dict:
        """Get statistics about the text corpus."""
        total_texts = len(texts)
        total_words = sum(len(text.split()) for text in texts)
        avg_words_per_text = total_words / total_texts if total_texts > 0 else 0
        
        empty_texts = sum(1 for text in texts if not text.strip())
        
        return {
            'total_texts': total_texts,
            'total_words': total_words,
            'average_words_per_text': avg_words_per_text,
            'empty_texts': empty_texts,
            'empty_text_percentage': (empty_texts / total_texts) * 100 if total_texts > 0 else 0
        } 