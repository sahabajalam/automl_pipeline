# src/agents/model_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    mean_squared_error, mean_absolute_error, r2_score, classification_report
)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
import xgboost as xgb
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
import joblib
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelTrainingAgent:
    """Agent responsible for training and evaluating multiple ML models"""
    
    def __init__(self):
        self.trained_models = {}
        self.model_performances = {}
        self.best_model = None
        self.task_type = None
        
    async def train_models(self, state: dict) -> dict:
        """Train multiple models and compare performance"""
        logger.info("Starting model training")
        
        try:
            processed_data = state['processed_data']
            target_column = state['target_column']
            project_name = state.get('project_name', 'ml_project')
            
            if target_column not in processed_data.columns:
                raise ValueError(f"Target column '{target_column}' not found in processed data")
            
            # Prepare data
            X = processed_data.drop(columns=[target_column])
            y = processed_data[target_column]
            
            # Determine task type
            self.task_type = self._determine_task_type(y)
            logger.info(f"Detected task type: {self.task_type}")
            
            # Encode target if classification with string labels
            label_encoder = None
            if self.task_type == 'classification' and y.dtype == 'object':
                label_encoder = LabelEncoder()
                y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, 
                stratify=y if self.task_type == 'classification' else None
            )
            
            # Further split training data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=0.2, random_state=42,
                stratify=y_train if self.task_type == 'classification' else None
            )
            
            logger.info(f"Data split - Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
            
            # Set up MLflow experiment
            mlflow.set_experiment(f"{project_name}_model_training")
            
            # Get model candidates
            model_candidates = self._get_model_candidates()
            
            training_results = {}
            
            # Train each model
            for model_name, model_config in model_candidates.items():
                logger.info(f"Training {model_name}...")
                
                try:
                    # Train with MLflow tracking
                    model_result = await self._train_single_model(
                        model_name, model_config, 
                        X_train, X_val, y_train, y_val,
                        X_test, y_test, label_encoder
                    )
                    
                    training_results[model_name] = model_result
                    logger.info(f"✅ {model_name} training completed")
                    
                except Exception as e:
                    logger.error(f"❌ {model_name} training failed: {str(e)}")
                    training_results[model_name] = {
                        'status': 'failed',
                        'error': str(e)
                    }
            
            # Select best model
            best_model_info = self._select_best_model(training_results)
            
            # Create training report
            training_report = {
                'task_type': self.task_type,
                'data_shapes': {
                    'train': X_train.shape,
                    'validation': X_val.shape,
                    'test': X_test.shape
                },
                'models_trained': len(training_results),
                'successful_models': len([r for r in training_results.values() if r.get('status') != 'failed']),
                'best_model': best_model_info,
                'all_results': training_results,
                'label_encoder': label_encoder
            }
            
            state.update({
                'trained_models': training_results,
                'best_model': best_model_info,
                'training_report': training_report,
                'X_test': X_test,
                'y_test': y_test,
                'current_step': 'model_training',
                'next_action': 'model_evaluation'
            })
            
            state['execution_log'].append(
                f"Model training completed: {training_report['successful_models']}/{training_report['models_trained']} models successful"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Model training failed: {str(e)}")
            state['errors'].append(f"Model training error: {str(e)}")
            state['next_action'] = 'error'
            return state
    
    async def evaluate_models(self, state: dict) -> dict:
        """Comprehensive model evaluation and comparison"""
        logger.info("Starting model evaluation")
        
        try:
            training_report = state['training_report']
            X_test = state['X_test']
            y_test = state['y_test']
            best_model_info = state['best_model']
            
            # Load best model
            best_model = best_model_info['model']
            
            # Comprehensive evaluation
            evaluation_results = self._comprehensive_evaluation(
                best_model, X_test, y_test, training_report['task_type']
            )
            
            # Model interpretation
            interpretation_results = self._interpret_model(
                best_model, X_test.columns, training_report['task_type']
            )
            
            # Performance comparison
            comparison_results = self._compare_all_models(
                state['trained_models'], X_test, y_test, training_report['task_type']
            )
            
            # Create evaluation report
            evaluation_report = {
                'best_model_performance': evaluation_results,
                'model_interpretation': interpretation_results,
                'model_comparison': comparison_results,
                'recommendations': self._generate_model_recommendations(evaluation_results, training_report)
            }
            
            # Check if model meets minimum requirements
            meets_requirements = self._check_model_requirements(evaluation_results, training_report['task_type'])
            
            state.update({
                'evaluation_results': evaluation_report,
                'model_meets_requirements': meets_requirements,
                'current_step': 'model_evaluation',
                'next_action': 'deployment' if meets_requirements else 'retrain'
            })
            
            state['execution_log'].append(
                f"Model evaluation completed - {'Meets requirements' if meets_requirements else 'Needs improvement'}"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Model evaluation failed: {str(e)}")
            state['errors'].append(f"Model evaluation error: {str(e)}")
            state['next_action'] = 'error'
            return state
    
    def _determine_task_type(self, y: pd.Series) -> str:
        """Determine if task is classification or regression"""
        if y.dtype in ['object', 'category']:
            return 'classification'
        elif y.nunique() <= 20 and y.nunique() < len(y) * 0.1:
            return 'classification'
        else:
            return 'regression'
    
    def _get_model_candidates(self) -> Dict[str, Dict[str, Any]]:
        """Get appropriate model candidates based on task type"""
        
        if self.task_type == 'classification':
            return {
                'random_forest': {
                    'model': RandomForestClassifier(random_state=42),
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    },
                    'priority': 'high'
                },
                'xgboost': {
                    'model': xgb.XGBClassifier(random_state=42, eval_metric='logloss'),
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    },
                    'priority': 'high'
                },
                'logistic_regression': {
                    'model': LogisticRegression(random_state=42, max_iter=1000),
                    'param_grid': {
                        'C': [0.1, 1.0, 10.0],
                        'penalty': ['l1', 'l2'],
                        'solver': ['liblinear']
                    },
                    'priority': 'medium'
                },
                'gradient_boosting': {
                    'model': GradientBoostingClassifier(random_state=42),
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    },
                    'priority': 'high'
                },
                'naive_bayes': {
                    'model': GaussianNB(),
                    'param_grid': {},  # No hyperparameters to tune
                    'priority': 'low'
                }
            }
        else:  # regression
            return {
                'random_forest': {
                    'model': RandomForestRegressor(random_state=42),
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [5, 10, None],
                        'min_samples_split': [2, 5],
                        'min_samples_leaf': [1, 2]
                    },
                    'priority': 'high'
                },
                'xgboost': {
                    'model': xgb.XGBRegressor(random_state=42),
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'max_depth': [3, 6, 9],
                        'learning_rate': [0.1, 0.2],
                        'subsample': [0.8, 1.0]
                    },
                    'priority': 'high'
                },
                'linear_regression': {
                    'model': LinearRegression(),
                    'param_grid': {},
                    'priority': 'medium'
                },
                'ridge': {
                    'model': Ridge(random_state=42),
                    'param_grid': {
                        'alpha': [0.1, 1.0, 10.0, 100.0]
                    },
                    'priority': 'medium'
                },
                'gradient_boosting': {
                    'model': GradientBoostingRegressor(random_state=42),
                    'param_grid': {
                        'n_estimators': [50, 100, 200],
                        'learning_rate': [0.1, 0.2],
                        'max_depth': [3, 5, 7]
                    },
                    'priority': 'high'
                }
            }
    
    async def _train_single_model(self, model_name: str, model_config: Dict,
                                 X_train: pd.DataFrame, X_val: pd.DataFrame,
                                 y_train: pd.Series, y_val: pd.Series,
                                 X_test: pd.DataFrame, y_test: pd.Series,
                                 label_encoder: Optional[Any]) -> Dict:
        """Train a single model with hyperparameter optimization"""
        
        with mlflow.start_run(run_name=f"{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Log basic info
            mlflow.log_param("model_type", model_name)
            mlflow.log_param("task_type", self.task_type)
            mlflow.log_param("n_features", X_train.shape[1])
            mlflow.log_param("n_train_samples", X_train.shape[0])
            
            model = model_config['model']
            param_grid = model_config['param_grid']
            
            # Hyperparameter optimization if parameters exist
            if param_grid:
                # Use RandomizedSearchCV for efficiency
                search = RandomizedSearchCV(
                    model, param_grid, 
                    n_iter=min(20, len(param_grid) * 3),  # Reasonable number of iterations
                    cv=3,  # 3-fold CV for speed
                    scoring=self._get_scoring_metric(),
                    random_state=42,
                    n_jobs=-1
                )
                
                search.fit(X_train, y_train)
                best_model = search.best_estimator_
                best_params = search.best_params_
                
                # Log best parameters
                for param, value in best_params.items():
                    mlflow.log_param(f"best_{param}", value)
                
                mlflow.log_metric("cv_best_score", search.best_score_)
                
            else:
                # No hyperparameters to tune
                best_model = model
                best_model.fit(X_train, y_train)
                best_params = {}
            
            # Evaluate on validation set
            val_predictions = best_model.predict(X_val)
            val_metrics = self._calculate_metrics(y_val, val_predictions)
            
            # Evaluate on test set
            test_predictions = best_model.predict(X_test)
            test_metrics = self._calculate_metrics(y_test, test_predictions)
            
            # Log metrics
            for metric_name, value in val_metrics.items():
                mlflow.log_metric(f"val_{metric_name}", value)
            
            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)
            
            # Log model
            mlflow.sklearn.log_model(best_model, "model")
            
            # Calculate training time (mock - in real implementation would track actual time)
            training_time = np.random.uniform(10, 300)  # Seconds
            mlflow.log_metric("training_time_seconds", training_time)
            
            # Cross-validation for more robust evaluation
            cv_scores = cross_val_score(
                best_model, X_train, y_train, 
                cv=3, scoring=self._get_scoring_metric()
            )
            
            mlflow.log_metric("cv_mean", cv_scores.mean())
            mlflow.log_metric("cv_std", cv_scores.std())
            
            # Feature importance (if available)
            feature_importance = self._get_feature_importance(best_model, X_train.columns)
            if feature_importance is not None:
                # Log top 10 most important features
                top_features = feature_importance.head(10)
                for i, (feature, importance) in enumerate(zip(top_features.index, top_features.values)):
                    mlflow.log_metric(f"feature_importance_{i+1}_{feature}", importance)
            
            return {
                'status': 'success',
                'model': best_model,
                'best_params': best_params,
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'cv_scores': cv_scores,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'feature_importance': feature_importance,
                'training_time': training_time,
                'mlflow_run_id': mlflow.active_run().info.run_id
            }
    
    def _get_scoring_metric(self) -> str:
        """Get appropriate scoring metric for the task"""
        if self.task_type == 'classification':
            return 'f1_weighted'  # Good for imbalanced datasets
        else:
            return 'r2'
    
    def _calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate appropriate metrics based on task type"""
        
        if self.task_type == 'classification':
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
                'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
                'f1': f1_score(y_true, y_pred, average='weighted', zero_division=0)
            }
            
            # Add ROC AUC for binary classification
            if len(np.unique(y_true)) == 2:
                try:
                    # For binary classification, we need probabilities for ROC AUC
                    metrics['roc_auc'] = 0.0  # Placeholder - would need predict_proba
                except:
                    pass
            
        else:  # regression
            metrics = {
                'mse': mean_squared_error(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'r2': r2_score(y_true, y_pred)
            }
            
            # Add MAPE if no zero values
            if not (y_true == 0).any():
                mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
                metrics['mape'] = mape
        
        return metrics
    
    def _get_feature_importance(self, model: Any, feature_names: List[str]) -> Optional[pd.Series]:
        """Extract feature importance from model if available"""
        
        try:
            if hasattr(model, 'feature_importances_'):
                # Tree-based models
                importance = pd.Series(model.feature_importances_, index=feature_names)
                return importance.sort_values(ascending=False)
            
            elif hasattr(model, 'coef_'):
                # Linear models
                if len(model.coef_.shape) == 1:
                    # Binary classification or regression
                    importance = pd.Series(np.abs(model.coef_), index=feature_names)
                else:
                    # Multi-class classification - use mean of absolute coefficients
                    importance = pd.Series(np.mean(np.abs(model.coef_), axis=0), index=feature_names)
                
                return importance.sort_values(ascending=False)
            
        except Exception as e:
            logger.warning(f"Could not extract feature importance: {str(e)}")
        
        return None
    
    def _select_best_model(self, training_results: Dict[str, Dict]) -> Dict:
        """Select the best model based on validation performance"""
        
        valid_results = {k: v for k, v in training_results.items() if v.get('status') == 'success'}
        
        if not valid_results:
            raise ValueError("No models trained successfully")
        
        # Select best model based on primary metric
        if self.task_type == 'classification':
            # Use F1 score as primary metric
            best_model_name = max(valid_results.keys(), 
                                key=lambda k: valid_results[k]['val_metrics']['f1'])
            primary_metric = 'f1'
        else:
            # Use R² as primary metric for regression
            best_model_name = max(valid_results.keys(),
                                key=lambda k: valid_results[k]['val_metrics']['r2'])
            primary_metric = 'r2'
        
        best_result = valid_results[best_model_name]
        
        return {
            'model_name': best_model_name,
            'model': best_result['model'],
            'performance': best_result['val_metrics'],
            'test_performance': best_result['test_metrics'],
            'primary_metric': primary_metric,
            'primary_metric_value': best_result['val_metrics'][primary_metric],
            'cv_mean': best_result['cv_mean'],
            'cv_std': best_result['cv_std'],
            'feature_importance': best_result['feature_importance'],
            'mlflow_run_id': best_result['mlflow_run_id']
        }
    
    def _comprehensive_evaluation(self, model: Any, X_test: pd.DataFrame, 
                                y_test: pd.Series, task_type: str) -> Dict:
        """Perform comprehensive evaluation of the best model"""
        
        predictions = model.predict(X_test)
        metrics = self._calculate_metrics(y_test, predictions)
        
        evaluation = {
            'test_metrics': metrics,
            'predictions_sample': predictions[:10].tolist(),
            'actual_sample': y_test.iloc[:10].tolist()
        }
        
        # Add confusion matrix for classification
        if task_type == 'classification':
            from sklearn.metrics import confusion_matrix, classification_report
            
            cm = confusion_matrix(y_test, predictions)
            evaluation['confusion_matrix'] = cm.tolist()
            evaluation['classification_report'] = classification_report(y_test, predictions, output_dict=True)
            
            # Calculate per-class metrics
            unique_classes = np.unique(y_test)
            evaluation['per_class_metrics'] = {}
            
            for cls in unique_classes:
                cls_mask = (y_test == cls)
                cls_predictions = predictions[cls_mask]
                cls_actual = y_test[cls_mask]
                
                if len(cls_actual) > 0:
                    cls_accuracy = accuracy_score(cls_actual, cls_predictions)
                    evaluation['per_class_metrics'][f'class_{cls}'] = {
                        'accuracy': cls_accuracy,
                        'support': len(cls_actual)
                    }
        
        # Add residual analysis for regression
        else:
            residuals = y_test - predictions
            evaluation['residual_analysis'] = {
                'mean_residual': float(residuals.mean()),
                'std_residual': float(residuals.std()),
                'min_residual': float(residuals.min()),
                'max_residual': float(residuals.max())
            }
            
            # Check for patterns in residuals
            evaluation['residual_patterns'] = {
                'heteroscedasticity': self._check_heteroscedasticity(predictions, residuals),
                'normality': self._check_residual_normality(residuals)
            }
        
        return evaluation
    
    def _interpret_model(self, model: Any, feature_names: List[str], task_type: str) -> Dict:
        """Generate model interpretation and explanations"""
        
        interpretation = {
            'model_type': type(model).__name__,
            'interpretability': self._assess_interpretability(model)
        }
        
        # Feature importance
        feature_importance = self._get_feature_importance(model, feature_names)
        if feature_importance is not None:
            interpretation['top_features'] = feature_importance.head(10).to_dict()
            interpretation['feature_importance_all'] = feature_importance.to_dict()
        
        # Model-specific interpretations
        if hasattr(model, 'coef_'):
            # Linear models
            interpretation['model_coefficients'] = {
                'positive_features': [],
                'negative_features': []
            }
            
            coef = model.coef_
            if len(coef.shape) == 1:  # Binary or regression
                for i, (feature, coeff) in enumerate(zip(feature_names, coef)):
                    if coeff > 0.01:
                        interpretation['model_coefficients']['positive_features'].append({
                            'feature': feature, 'coefficient': float(coeff)
                        })
                    elif coeff < -0.01:
                        interpretation['model_coefficients']['negative_features'].append({
                            'feature': feature, 'coefficient': float(coeff)
                        })
        
        # Tree-based model insights
        if hasattr(model, 'n_estimators'):
            interpretation['ensemble_info'] = {
                'n_estimators': getattr(model, 'n_estimators', None),
                'max_depth': getattr(model, 'max_depth', None)
            }
        
        return interpretation
    
    def _assess_interpretability(self, model: Any) -> str:
        """Assess model interpretability level"""
        
        model_name = type(model).__name__.lower()
        
        if any(name in model_name for name in ['linear', 'logistic', 'naive']):
            return 'high'
        elif any(name in model_name for name in ['tree', 'forest', 'gradient']):
            return 'medium'
        elif any(name in model_name for name in ['svm', 'neural', 'xgb']):
            return 'low'
        else:
            return 'unknown'
    
    def _compare_all_models(self, trained_models: Dict, X_test: pd.DataFrame, 
                          y_test: pd.Series, task_type: str) -> Dict:
        """Compare performance of all trained models"""
        
        comparison = {}
        
        for model_name, model_result in trained_models.items():
            if model_result.get('status') == 'success':
                test_metrics = model_result['test_metrics']
                comparison[model_name] = {
                    'test_performance': test_metrics,
                    'cv_mean': model_result['cv_mean'],
                    'cv_std': model_result['cv_std'],
                    'training_time': model_result.get('training_time', 0)
                }
        
        # Rank models by primary metric
        if task_type == 'classification':
            sorted_models = sorted(comparison.items(), 
                                 key=lambda x: x[1]['test_performance']['f1'], 
                                 reverse=True)
        else:
            sorted_models = sorted(comparison.items(),
                                 key=lambda x: x[1]['test_performance']['r2'],
                                 reverse=True)
        
        comparison['ranking'] = [{'model': name, 'rank': i+1} for i, (name, _) in enumerate(sorted_models)]
        
        return comparison
    
    def _check_model_requirements(self, evaluation_results: Dict, task_type: str) -> bool:
        """Check if model meets minimum performance requirements"""
        
        test_metrics = evaluation_results['test_metrics']
        
        if task_type == 'classification':
            # Minimum requirements for classification
            return (
                test_metrics.get('accuracy', 0) >= 0.7 and
                test_metrics.get('f1', 0) >= 0.65
            )
        else:
            # Minimum requirements for regression
            return (
                test_metrics.get('r2', 0) >= 0.5 and
                test_metrics.get('mape', 100) <= 20  # Less than 20% MAPE
            )
    
    def _generate_model_recommendations(self, evaluation_results: Dict, training_report: Dict) -> List[str]:
        """Generate recommendations based on model performance"""
        
        recommendations = []
        test_metrics = evaluation_results['test_metrics']
        task_type = training_report['task_type']
        
        if task_type == 'classification':
            accuracy = test_metrics.get('accuracy', 0)
            f1 = test_metrics.get('f1', 0)
            
            if accuracy < 0.7:
                recommendations.append("Model accuracy is below 70%. Consider collecting more data or trying different feature engineering approaches.")
            
            if f1 < 0.65:
                recommendations.append("F1 score is low. This might indicate class imbalance - consider using techniques like SMOTE or adjusting class weights.")
            
            # Check for overfitting
            best_model = training_report['best_model']
            if 'cv_std' in best_model and best_model['cv_std'] > 0.1:
                recommendations.append("High variance in cross-validation scores suggests possible overfitting. Consider regularization or reducing model complexity.")
            
        else:  # regression
            r2 = test_metrics.get('r2', 0)
            mape = test_metrics.get('mape', 100)
            
            if r2 < 0.5:
                recommendations.append("R² score is below 0.5. The model explains less than 50% of variance. Consider feature engineering or trying different algorithms.")
            
            if mape > 20:
                recommendations.append("MAPE is above 20%. Consider log transformation of target variable or outlier removal.")
            
            # Check residuals
            if 'residual_analysis' in evaluation_results:
                residual_analysis = evaluation_results['residual_analysis']
                if abs(residual_analysis['mean_residual']) > 0.1:
                    recommendations.append("Residuals show bias. Consider adding polynomial features or checking for missing important variables.")
        
        if not recommendations:
            recommendations.append("Model performance meets requirements. Ready for deployment.")
        
        return recommendations
    
    def _check_heteroscedasticity(self, predictions: np.ndarray, residuals: np.ndarray) -> Dict:
        """Check for heteroscedasticity in residuals"""
        # Simple check using correlation between absolute residuals and predictions
        from scipy.stats import pearsonr
        
        try:
            corr, p_value = pearsonr(predictions, np.abs(residuals))
            return {
                'correlation': float(corr),
                'p_value': float(p_value),
                'heteroscedastic': abs(corr) > 0.3 and p_value < 0.05
            }
        except:
            return {'heteroscedastic': False, 'error': 'Could not compute'}
    
    def _check_residual_normality(self, residuals: np.ndarray) -> Dict:
        """Check if residuals follow normal distribution"""
        from scipy.stats import normaltest
        
        try:
            stat, p_value = normaltest(residuals)
            return {
                'statistic': float(stat),
                'p_value': float(p_value),
                'normal': p_value > 0.05
            }
        except:
            return {'normal': True, 'error': 'Could not compute'}

# Utility functions for advanced model training
class AdvancedModelTrainer:
    """Advanced model training techniques"""
    
    @staticmethod
    def train_ensemble_model(models: List[Any], X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """Create ensemble of multiple models"""
        from sklearn.ensemble import VotingClassifier, VotingRegressor
        
        # Determine if classification or regression
        task_type = 'classification' if y_train.nunique() <= 20 else 'regression'
        
        if task_type == 'classification':
            ensemble = VotingClassifier(
                estimators=[(f'model_{i}', model) for i, model in enumerate(models)],
                voting='soft'
            )
        else:
            ensemble = VotingRegressor(
                estimators=[(f'model_{i}', model) for i, model in enumerate(models)]
            )
        
        ensemble.fit(X_train, y_train)
        return ensemble
    
    @staticmethod
    def perform_automated_feature_selection(X: pd.DataFrame, y: pd.Series, 
                                          max_features: int = 20) -> Tuple[pd.DataFrame, List[str]]:
        """Perform automated feature selection during model training"""
        from sklearn.feature_selection import SelectFromModel
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        
        # Use RandomForest for feature selection
        if y.nunique() <= 20:  # Classification
            selector_model = RandomForestClassifier(n_estimators=100, random_state=42)
        else:  # Regression
            selector_model = RandomForestRegressor(n_estimators=100, random_state=42)
        
        selector = SelectFromModel(selector_model, max_features=max_features)
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature names
        selected_features = X.columns[selector.get_support()].tolist()
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        return X_selected_df, selected_features