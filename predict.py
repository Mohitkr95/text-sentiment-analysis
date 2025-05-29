#!/usr/bin/env python3
"""
Sentiment Prediction Script
============================

This script loads trained models and makes sentiment predictions on new text.

Usage:
    python predict.py "This restaurant is amazing!"
    python predict.py --file reviews.txt
"""

import sys
import argparse
import pickle
from pathlib import Path
import pandas as pd

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

from config.config import MODELS_DIR
from src.preprocessing.text_processor import TextProcessor
from src.preprocessing.feature_extractor import FeatureExtractor

class SentimentPredictor:
    """Simple sentiment prediction class."""
    
    def __init__(self, model_path=None, feature_extractor_path=None):
        """Initialize the predictor with trained models."""
        self.text_processor = TextProcessor()
        
        # Default paths
        if model_path is None:
            model_files = list(MODELS_DIR.glob("best_model_*.pkl"))
            if model_files:
                model_path = model_files[0]
            else:
                raise FileNotFoundError("No trained model found. Please run main.py first.")
        
        if feature_extractor_path is None:
            feature_extractor_path = MODELS_DIR / "feature_extractor.pkl"
        
        # Load models
        print(f"Loading model from: {model_path}")
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        print(f"Loading feature extractor from: {feature_extractor_path}")
        self.feature_extractor = FeatureExtractor.load(str(feature_extractor_path))
        
        print("Models loaded successfully!")
    
    def predict_single(self, text):
        """Predict sentiment for a single text."""
        # Preprocess text
        processed_text = self.text_processor.process_corpus([text])
        
        # Extract features
        features = self.feature_extractor.transform(processed_text)
        
        # Make prediction
        prediction = self.model.predict(features)[0]
        
        # Get probability if available
        try:
            probability = self.model.predict_proba(features)[0]
            confidence = max(probability)
        except:
            confidence = None
        
        return {
            'text': text,
            'prediction': 'Positive' if prediction == 1 else 'Negative',
            'confidence': confidence,
            'processed_text': processed_text[0]
        }
    
    def predict_batch(self, texts):
        """Predict sentiment for multiple texts."""
        # Preprocess texts
        processed_texts = self.text_processor.process_corpus(texts)
        
        # Extract features
        features = self.feature_extractor.transform(processed_texts)
        
        # Make predictions
        predictions = self.model.predict(features)
        
        # Get probabilities if available
        try:
            probabilities = self.model.predict_proba(features)
            confidences = [max(prob) for prob in probabilities]
        except:
            confidences = [None] * len(texts)
        
        results = []
        for i, text in enumerate(texts):
            results.append({
                'text': text,
                'prediction': 'Positive' if predictions[i] == 1 else 'Negative',
                'confidence': confidences[i],
                'processed_text': processed_texts[i]
            })
        
        return results

def main():
    parser = argparse.ArgumentParser(description="Predict sentiment of text")
    parser.add_argument('text', nargs='?', help='Text to analyze')
    parser.add_argument('--file', '-f', help='File containing texts to analyze (one per line)')
    parser.add_argument('--model', '-m', help='Path to trained model file')
    parser.add_argument('--extractor', '-e', help='Path to feature extractor file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    if not args.text and not args.file:
        parser.print_help()
        return
    
    try:
        # Initialize predictor
        predictor = SentimentPredictor(args.model, args.extractor)
        
        if args.text:
            # Single text prediction
            result = predictor.predict_single(args.text)
            
            print(f"\nText: {result['text']}")
            print(f"Sentiment: {result['prediction']}")
            if result['confidence']:
                print(f"Confidence: {result['confidence']:.3f}")
            
            if args.verbose:
                print(f"Processed: {result['processed_text']}")
        
        elif args.file:
            # Batch prediction from file
            with open(args.file, 'r', encoding='utf-8') as f:
                texts = [line.strip() for line in f if line.strip()]
            
            print(f"Analyzing {len(texts)} texts from {args.file}...")
            results = predictor.predict_batch(texts)
            
            # Display results
            print("\nResults:")
            print("-" * 80)
            for i, result in enumerate(results, 1):
                print(f"{i}. Text: {result['text'][:100]}{'...' if len(result['text']) > 100 else ''}")
                print(f"   Sentiment: {result['prediction']}")
                if result['confidence']:
                    print(f"   Confidence: {result['confidence']:.3f}")
                if args.verbose:
                    print(f"   Processed: {result['processed_text']}")
                print()
            
            # Summary
            positive_count = sum(1 for r in results if r['prediction'] == 'Positive')
            negative_count = len(results) - positive_count
            print(f"Summary: {positive_count} positive, {negative_count} negative out of {len(results)} reviews")
            
            # Save results if batch
            if len(results) > 1:
                output_file = Path(args.file).stem + "_predictions.csv"
                df = pd.DataFrame(results)
                df.to_csv(output_file, index=False)
                print(f"Results saved to: {output_file}")
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()