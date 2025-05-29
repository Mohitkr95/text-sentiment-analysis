#!/usr/bin/env python3
"""
Advanced Sentiment Analysis Pipeline
====================================

This script implements a comprehensive sentiment analysis system with:
- Advanced text preprocessing
- Feature extraction with TF-IDF and additional features
- Multiple machine learning models with ensemble methods
- Hyperparameter tuning
- Class imbalance handling
- Comprehensive evaluation and logging

"""

import sys
import os
import warnings
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent
sys.path.append(str(PROJECT_ROOT))

# Import custom modules
from config.config import *
from src.utils.logger import setup_logger
from src.preprocessing.text_processor import TextProcessor
from src.preprocessing.feature_extractor import FeatureExtractor
from src.models.ensemble_models import EnsembleModels
from src.evaluation.evaluator import ModelEvaluator

def load_and_explore_data(logger):
    """Load and explore the dataset."""
    logger.info("Loading and exploring dataset...")
    
    # Load dataset
    try:
        dataset = pd.read_csv(DATASET_PATH, delimiter='\t', quoting=3)
        logger.info(f"Dataset loaded successfully: {dataset.shape}")
    except Exception as e:
        logger.error(f"Error loading dataset: {e}")
        raise
    
    # Basic exploration
    logger.info("Dataset Info:")
    logger.info(f"  Shape: {dataset.shape}")
    logger.info(f"  Columns: {list(dataset.columns)}")
    logger.info(f"  Missing values: {dataset.isnull().sum().to_dict()}")
    
    # Class distribution
    if 'Liked' in dataset.columns:
        class_dist = dataset['Liked'].value_counts()
        logger.info(f"  Class distribution: {class_dist.to_dict()}")
        logger.info(f"  Class balance ratio: {class_dist.min()/class_dist.max():.3f}")
    
    # Sample data
    logger.info("Sample reviews:")
    for i in range(min(3, len(dataset))):
        logger.info(f"  Review {i+1}: {dataset.iloc[i]['Review'][:100]}...")
        logger.info(f"  Sentiment: {dataset.iloc[i]['Liked']}")
    
    return dataset

def main():
    """Main pipeline execution."""
    # Setup logging
    logger = setup_logger("SentimentAnalysis")
    logger.info("Starting Advanced Sentiment Analysis Pipeline")
    logger.info("=" * 60)
    
    try:
        # 1. Load and explore data
        dataset = load_and_explore_data(logger)
        
        # Extract texts and labels
        texts = dataset['Review'].tolist()
        labels = dataset['Liked'].values
        
        # 2. Text preprocessing
        logger.info("Initializing text processor...")
        text_processor = TextProcessor()
        
        # Get text statistics before processing
        stats_before = text_processor.get_text_statistics(texts)
        logger.info(f"Text statistics before processing: {stats_before}")
        
        # Process texts
        processed_texts = text_processor.process_corpus(texts, correct_spelling=False)
        
        # Get text statistics after processing
        stats_after = text_processor.get_text_statistics(processed_texts)
        logger.info(f"Text statistics after processing: {stats_after}")
        
        # 3. Feature extraction
        logger.info("Initializing feature extractor...")
        feature_extractor = FeatureExtractor(use_tfidf=True, use_additional_features=True)
        
        # Split data first for proper feature selection
        X_temp, X_test, y_temp, y_test = train_test_split(
            processed_texts, labels, 
            test_size=TEST_SIZE, 
            random_state=RANDOM_STATE, 
            stratify=labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=VALIDATION_SIZE,
            random_state=RANDOM_STATE,
            stratify=y_temp
        )
        
        logger.info(f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
        
        # Extract features
        X_train_features = feature_extractor.fit_transform(X_train, y_train)
        X_val_features = feature_extractor.transform(X_val)
        X_test_features = feature_extractor.transform(X_test)
        
        logger.info(f"Feature extraction completed: {X_train_features.shape[1]} features")
        
        # 4. Model training and ensemble
        logger.info("Initializing ensemble models...")
        ensemble = EnsembleModels(handle_imbalance=True)
        
        # Train individual models
        individual_scores = ensemble.train_individual_models(
            X_train_features, y_train, X_val_features, y_val, 
            tune_hyperparameters=True
        )
        
        logger.info("Individual model scores:")
        for model, score in individual_scores.items():
            logger.info(f"  {model}: {score:.4f}")
        
        # Create ensemble models
        ensemble_scores = ensemble.create_ensemble_models(
            X_train_features, y_train, X_val_features, y_val
        )
        
        logger.info("Ensemble model scores:")
        for model, score in ensemble_scores.items():
            logger.info(f"  {model}: {score:.4f}")
        
        # Get model rankings
        rankings = ensemble.get_model_rankings(X_val_features, y_val)
        logger.info("Model rankings (by F1 score):")
        for i, (model, score) in enumerate(rankings, 1):
            logger.info(f"  {i}. {model}: {score:.4f}")
        
        # 5. Final evaluation
        logger.info("Performing final evaluation on test set...")
        evaluator = ModelEvaluator(save_plots=True)
        
        # Evaluate best model
        best_model_name = rankings[0][0]
        logger.info(f"Best model: {best_model_name}")
        
        # Get the best model
        best_model = ensemble.models[best_model_name]
        
        # Comprehensive evaluation
        evaluation_results = evaluator.evaluate_model_comprehensive(
            best_model, X_test_features, y_test, best_model_name
        )
        
        # Print detailed report
        print(evaluation_results['detailed_report'])
        
        # Cross-validation on full dataset
        logger.info("Performing cross-validation on full dataset...")
        X_full_features = feature_extractor.fit_transform(processed_texts, labels)
        cv_results = ensemble.cross_validate_best_model(X_full_features, labels)
        
        # Create model comparison visualization
        all_scores = {**individual_scores, **ensemble_scores}
        evaluator.plot_model_comparison(
            {name: {'f1_score': score} for name, score in all_scores.items()},
            metric='f1_score',
            title="Model Performance Comparison"
        )
        
        # Create interactive dashboard
        model_results = {}
        for name, score in all_scores.items():
            model_results[name] = {
                'f1_score': score,
                'accuracy': 0.0  # Placeholder - would need to calculate properly
            }
        
        dashboard = evaluator.create_interactive_dashboard(
            model_results,
            y_test,
            evaluation_results['predictions'],
            evaluation_results['probabilities']
        )
        
        # 6. Save models and results
        logger.info("Saving models and results...")
        
        # Save best model
        best_model_path = MODELS_DIR / f"best_model_{best_model_name}.pkl"
        ensemble.save_best_model(str(best_model_path))
        
        # Save feature extractor
        feature_extractor_path = MODELS_DIR / "feature_extractor.pkl"
        feature_extractor.save(str(feature_extractor_path))
        
        # Save results summary
        results_summary = {
            'best_model': best_model_name,
            'best_score': rankings[0][1],
            'all_scores': all_scores,
            'cv_results': cv_results,
            'test_metrics': evaluation_results['metrics'],
            'feature_count': X_train_features.shape[1],
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
        
        results_df = pd.DataFrame([results_summary])
        results_path = LOGS_DIR / "results_summary.csv"
        results_df.to_csv(results_path, index=False)
        
        # 7. Final summary
        logger.info("=" * 60)
        logger.info("FINAL RESULTS SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Best Model: {best_model_name}")
        logger.info(f"Best Validation F1 Score: {rankings[0][1]:.4f}")
        logger.info(f"Test F1 Score: {evaluation_results['metrics']['f1_score']:.4f}")
        logger.info(f"Test Accuracy: {evaluation_results['metrics']['accuracy']:.4f}")
        logger.info(f"Cross-validation F1: {cv_results['mean_f1']:.4f} ± {cv_results['std_f1']:.4f}")
        
        if 'roc_auc' in evaluation_results['metrics']:
            logger.info(f"Test ROC AUC: {evaluation_results['metrics']['roc_auc']:.4f}")
        
        logger.info(f"Total Features Used: {X_train_features.shape[1]}")
        logger.info(f"Models Saved to: {MODELS_DIR}")
        logger.info(f"Logs Saved to: {LOGS_DIR}")
        logger.info("=" * 60)
        
        # Display plots
        plt.show()
        
        logger.info("Pipeline completed successfully!")
        
        return {
            'best_model': ensemble,
            'feature_extractor': feature_extractor,
            'text_processor': text_processor,
            'evaluator': evaluator,
            'results': results_summary
        }
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}")
        raise

if __name__ == "__main__":
    results = main() 