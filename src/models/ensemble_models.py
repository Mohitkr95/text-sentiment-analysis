import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import pickle
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))
from config.config import CROSS_VALIDATION_FOLDS, RANDOM_STATE
from src.utils.logger import setup_logger, log_model_performance

class EnsembleModels:
    """Advanced ensemble models for sentiment analysis."""
    
    def __init__(self, handle_imbalance: bool = True):
        self.logger = setup_logger(self.__class__.__name__)
        self.handle_imbalance = handle_imbalance
        self.models = {}
        self.best_model = None
        self.best_score = 0.0
        self.smote = SMOTE(random_state=RANDOM_STATE) if handle_imbalance else None
        
    def _get_base_models(self) -> Dict[str, Any]:
        """Get base models for ensemble."""
        base_models = {
            'logistic_regression': LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight='balanced' if self.handle_imbalance else None
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                class_weight='balanced' if self.handle_imbalance else None,
                n_jobs=-1
            ),
            'svm': SVC(
                probability=True,
                random_state=RANDOM_STATE,
                class_weight='balanced' if self.handle_imbalance else None
            ),
            'multinomial_nb': MultinomialNB(),
            'knn': KNeighborsClassifier(
                n_neighbors=5,
                n_jobs=-1
            )
        }
        
        if self.handle_imbalance:
            base_models['balanced_rf'] = BalancedRandomForestClassifier(
                n_estimators=100,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
            base_models['balanced_bagging'] = BalancedBaggingClassifier(
                base_estimator=DecisionTreeClassifier(),
                n_estimators=50,
                random_state=RANDOM_STATE,
                n_jobs=-1
            )
        
        return base_models
    
    def _get_hyperparameter_grids(self) -> Dict[str, Dict]:
        """Get hyperparameter grids for tuning."""
        param_grids = {
            'logistic_regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'random_forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10]
            },
            'svm': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            },
            'multinomial_nb': {
                'alpha': [0.1, 0.5, 1.0, 2.0]
            },
            'knn': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
        }
        
        if self.handle_imbalance:
            param_grids['balanced_rf'] = {
                'n_estimators': [50, 100],
                'max_depth': [10, 20, None]
            }
            param_grids['balanced_bagging'] = {
                'n_estimators': [30, 50],
                'max_samples': [0.5, 0.8, 1.0]
            }
        
        return param_grids
    
    def train_individual_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray,
                              tune_hyperparameters: bool = True) -> Dict[str, float]:
        """Train individual models with optional hyperparameter tuning."""
        self.logger.info("Training individual models...")
        
        # Apply SMOTE if handling imbalance
        if self.smote is not None:
            self.logger.info("Applying SMOTE to handle class imbalance...")
            X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train, y_train)
            self.logger.info(f"Original training set size: {len(y_train)}")
            self.logger.info(f"Balanced training set size: {len(y_train_balanced)}")
        else:
            X_train_balanced, y_train_balanced = X_train, y_train
        
        base_models = self._get_base_models()
        param_grids = self._get_hyperparameter_grids()
        scores = {}
        
        for name, model in base_models.items():
            self.logger.info(f"Training {name}...")
            
            try:
                if tune_hyperparameters and name in param_grids:
                    # Hyperparameter tuning
                    grid_search = GridSearchCV(
                        model,
                        param_grids[name],
                        cv=3,  # Reduced for speed
                        scoring='f1',
                        n_jobs=-1,
                        verbose=0
                    )
                    grid_search.fit(X_train_balanced, y_train_balanced)
                    best_model = grid_search.best_estimator_
                    self.logger.info(f"Best params for {name}: {grid_search.best_params_}")
                else:
                    # Train with default parameters
                    best_model = model
                    best_model.fit(X_train_balanced, y_train_balanced)
                
                # Evaluate on validation set
                y_pred = best_model.predict(X_val)
                score = f1_score(y_val, y_pred)
                scores[name] = score
                
                # Store the model
                self.models[name] = best_model
                
                # Update best model if this one is better
                if score > self.best_score:
                    self.best_score = score
                    self.best_model = best_model
                
                # Log performance
                metrics = {
                    'accuracy': accuracy_score(y_val, y_pred),
                    'f1_score': score
                }
                log_model_performance(self.logger, name, metrics)
                
            except Exception as e:
                self.logger.error(f"Error training {name}: {str(e)}")
                scores[name] = 0.0
        
        return scores
    
    def create_ensemble_models(self, X_train: np.ndarray, y_train: np.ndarray,
                              X_val: np.ndarray, y_val: np.ndarray) -> Dict[str, float]:
        """Create and train ensemble models."""
        self.logger.info("Creating ensemble models...")
        
        if len(self.models) < 2:
            self.logger.warning("Need at least 2 trained models for ensemble")
            return {}
        
        # Select top models for ensemble (exclude worst performers)
        model_scores = {name: f1_score(y_val, model.predict(X_val)) 
                       for name, model in self.models.items()}
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        top_models = sorted_models[:5]  # Top 5 models
        
        self.logger.info(f"Using top {len(top_models)} models for ensemble")
        
        # Create voting classifier
        estimators = [(name, self.models[name]) for name, _ in top_models]
        
        # Hard voting ensemble
        hard_voting = VotingClassifier(estimators=estimators, voting='hard')
        hard_voting.fit(X_train, y_train)
        hard_pred = hard_voting.predict(X_val)
        hard_score = f1_score(y_val, hard_pred)
        
        # Soft voting ensemble (if all models support probability prediction)
        try:
            soft_voting = VotingClassifier(estimators=estimators, voting='soft')
            soft_voting.fit(X_train, y_train)
            soft_pred = soft_voting.predict(X_val)
            soft_score = f1_score(y_val, soft_pred)
        except:
            self.logger.warning("Some models don't support probability prediction. Skipping soft voting.")
            soft_score = 0.0
            soft_voting = None
        
        # Store ensemble models
        self.models['hard_voting'] = hard_voting
        if soft_voting is not None:
            self.models['soft_voting'] = soft_voting
        
        # Update best model if ensemble is better
        if hard_score > self.best_score:
            self.best_score = hard_score
            self.best_model = hard_voting
        
        if soft_score > self.best_score:
            self.best_score = soft_score
            self.best_model = soft_voting
        
        # Log ensemble performance
        ensemble_scores = {'hard_voting': hard_score}
        if soft_voting is not None:
            ensemble_scores['soft_voting'] = soft_score
        
        for name, score in ensemble_scores.items():
            metrics = {
                'f1_score': score,
                'accuracy': accuracy_score(y_val, self.models[name].predict(X_val))
            }
            log_model_performance(self.logger, name, metrics)
        
        return ensemble_scores
    
    def get_model_rankings(self, X_val: np.ndarray, y_val: np.ndarray) -> List[Tuple[str, float]]:
        """Get ranking of all models by performance."""
        scores = {}
        for name, model in self.models.items():
            try:
                y_pred = model.predict(X_val)
                scores[name] = f1_score(y_val, y_pred)
            except:
                scores[name] = 0.0
        
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def predict(self, X: np.ndarray, use_best_model: bool = True) -> np.ndarray:
        """Make predictions using the best model or ensemble."""
        if use_best_model and self.best_model is not None:
            return self.best_model.predict(X)
        elif 'soft_voting' in self.models:
            return self.models['soft_voting'].predict(X)
        elif 'hard_voting' in self.models:
            return self.models['hard_voting'].predict(X)
        else:
            raise ValueError("No trained models available")
    
    def predict_proba(self, X: np.ndarray, use_best_model: bool = True) -> np.ndarray:
        """Get prediction probabilities."""
        if use_best_model and self.best_model is not None:
            if hasattr(self.best_model, 'predict_proba'):
                return self.best_model.predict_proba(X)
        elif 'soft_voting' in self.models:
            return self.models['soft_voting'].predict_proba(X)
        
        # Fallback to decision function if available
        if hasattr(self.best_model, 'decision_function'):
            decision = self.best_model.decision_function(X)
            # Convert to probabilities using sigmoid
            proba = 1 / (1 + np.exp(-decision))
            return np.column_stack([1 - proba, proba])
        
        raise ValueError("Model doesn't support probability prediction")
    
    def save_best_model(self, filepath: str):
        """Save the best performing model."""
        if self.best_model is None:
            raise ValueError("No best model to save")
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        self.logger.info(f"Best model saved to {filepath}")
    
    def cross_validate_best_model(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Perform cross-validation on the best model."""
        if self.best_model is None:
            raise ValueError("No best model available")
        
        cv_scores = cross_val_score(
            self.best_model, X, y, 
            cv=CROSS_VALIDATION_FOLDS, 
            scoring='f1',
            n_jobs=-1
        )
        
        cv_results = {
            'mean_f1': cv_scores.mean(),
            'std_f1': cv_scores.std(),
            'cv_scores': cv_scores.tolist()
        }
        
        self.logger.info(f"Cross-validation F1 score: {cv_results['mean_f1']:.4f} (+/- {cv_results['std_f1'] * 2:.4f})")
        
        return cv_results 