# tests/test_model_agent.py
import pytest
import pandas as pd
import numpy as np
import mlflow
from unittest.unfitted import mock
from src.agents.model_agent import ModelTrainingAgent, AdvancedModelTrainer
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

class TestModelTrainingAgent:
    
    @pytest.fixture
    def classification_data(self):
        """Create classification dataset for testing"""
        X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)
        
        feature_names = [f'feature_{i}' for i in range(10)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def regression_data(self):
        """Create regression dataset for testing"""
        X, y = make_regression(n_samples=500, n_features=8, noise=0.1, random_state=42)
        
        feature_names = [f'feature_{i}' for i in range(8)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def multiclass_data(self):
        """Create multi-class classification dataset for testing"""
        X, y = make_classification(n_samples=600, n_features=12, n_classes=3, random_state=42)
        
        feature_names = [f'feature_{i}' for i in range(12)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow for testing"""
        with mock.patch('mlflow.set_experiment'), \
             mock.patch('mlflow.start_run'), \
             mock.patch('mlflow.log_param'), \
             mock.patch('mlflow.log_metric'), \
             mock.patch('mlflow.sklearn.log_model'), \
             mock.patch('mlflow.active_run') as mock_run:
            
            mock_run.return_value.info.run_id = 'test_run_id_123'
            yield
    
    def test_task_type_determination(self, classification_data, regression_data):
        """Test automatic task type determination"""
        agent = ModelTrainingAgent()
        
        # Test classification detection
        y_class = classification_data['target']
        task_type = agent._determine_task_type(y_class)
        assert task_type == 'classification'
        
        # Test regression detection
        y_reg = regression_data['target']
        task_type = agent._determine_task_type(y_reg)
        assert task_type == 'regression'
        
        # Test string classification
        y_string = pd.Series(['cat', 'dog', 'cat', 'dog', 'bird'] * 20)
        task_type = agent._determine_task_type(y_string)
        assert task_type == 'classification'
    
    def test_model_candidates_generation(self):
        """Test model candidate generation for different task types"""
        agent = ModelTrainingAgent()
        
        # Test classification candidates
        agent.task_type = 'classification'
        candidates = agent._get_model_candidates()
        
        assert 'random_forest' in candidates
        assert 'xgboost' in candidates
        assert 'logistic_regression' in candidates
        
        # Check structure
        for model_name, config in candidates.items():
            assert 'model' in config
            assert 'param_grid' in config
            assert 'priority' in config
        
        # Test regression candidates
        agent.task_type = 'regression'
        candidates = agent._get_model_candidates()
        
        assert 'random_forest' in candidates
        assert 'xgboost' in candidates
        assert 'linear_regression' in candidates
    
    def test_metrics_calculation(self, classification_data, regression_data):
        """Test metrics calculation for different task types"""
        agent = ModelTrainingAgent()
        
        # Test classification metrics
        agent.task_type = 'classification'
        y_true = np.array([0, 1, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 0])
        
        metrics = agent._calculate_metrics(pd.Series(y_true), y_pred)
        
        assert 'accuracy' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        assert 0 <= metrics['accuracy'] <= 1
        
        # Test regression metrics
        agent.task_type = 'regression'
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 3.8, 5.2])
        
        metrics = agent._calculate_metrics(pd.Series(y_true), y_pred)
        
        assert 'mse' in metrics
        assert 'rmse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        assert metrics['rmse'] == np.sqrt(metrics['mse'])
    
    def test_feature_importance_extraction(self):
        """Test feature importance extraction from different model types"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        agent = ModelTrainingAgent()
        
        # Create sample data
        X = pd.DataFrame({
            'feature_1': [1, 2, 3, 4, 5],
            'feature_2': [2, 4, 6, 8, 10],
            'feature_3': [1, 1, 2, 2, 3]
        })
        y = pd.Series([0, 0, 1, 1, 1])
        
        # Test tree-based model (has feature_importances_)
        rf_model = RandomForestClassifier(n_estimators=10, random_state=42)
        rf_model.fit(X, y)
        
        importance = agent._get_feature_importance(rf_model, X.columns)
        assert importance is not None
        assert len(importance) == 3
        assert all(importance >= 0)
        
        # Test linear model (has coef_)
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X, y)
        
        importance = agent._get_feature_importance(lr_model, X.columns)
        assert importance is not None
        assert len(importance) == 3
    
    @pytest.mark.asyncio
    async def test_model_training_classification(self, classification_data, mock_mlflow):
        """Test complete model training pipeline for classification"""
        agent = ModelTrainingAgent()
        
        state = {
            'processed_data': classification_data,
            'target_column': 'target',
            'project_name': 'test_classification',
            'execution_log': []
        }
        
        result = await agent.train_models(state)
        
        # Check that training completed successfully
        assert 'trained_models' in result
        assert 'best_model' in result
        assert 'training_report' in result
        assert result['current_step'] == 'model_training'
        assert result['next_action'] == 'model_evaluation'
        
        # Check training report structure
        training_report = result['training_report']
        assert training_report['task_type'] == 'classification'
        assert 'models_trained' in training_report
        assert 'successful_models' in training_report
        assert 'best_model' in training_report
        
        # Check best model structure
        best_model = result['best_model']
        assert 'model_name' in best_model
        assert 'model' in best_model
        assert 'performance' in best_model
    
    @pytest.mark.asyncio
    async def test_model_training_regression(self, regression_data, mock_mlflow):
        """Test complete model training pipeline for regression"""
        agent = ModelTrainingAgent()
        
        state = {
            'processed_data': regression_data,
            'target_column': 'target',
            'project_name': 'test_regression',
            'execution_log': []
        }
        
        result = await agent.train_models(state)
        
        assert 'trained_models' in result
        assert 'best_model' in result
        assert result['training_report']['task_type'] == 'regression'
    
    @pytest.mark.asyncio
    async def test_model_evaluation(self, classification_data, mock_mlflow):
        """Test model evaluation pipeline"""
        agent = ModelTrainingAgent()
        
        # First train models
        state = {
            'processed_data': classification_data,
            'target_column': 'target',
            'project_name': 'test_evaluation',
            'execution_log': []
        }
        
        trained_state = await agent.train_models(state)
        
        # Then evaluate
        evaluated_state = await agent.evaluate_models(trained_state)
        
        assert 'evaluation_results' in evaluated_state
        assert 'model_meets_requirements' in evaluated_state
        assert evaluated_state['current_step'] == 'model_evaluation'
        
        # Check evaluation structure
        evaluation_results = evaluated_state['evaluation_results']
        assert 'best_model_performance' in evaluation_results
        assert 'model_interpretation' in evaluation_results
        assert 'model_comparison' in evaluation_results
        assert 'recommendations' in evaluation_results
    
    def test_model_selection(self, classification_data):
        """Test best model selection logic"""
        agent = ModelTrainingAgent()
        agent.task_type = 'classification'
        
        # Mock training results
        training_results = {
            'model_a': {
                'status': 'success',
                'model': 'mock_model_a',
                'val_metrics': {'f1': 0.8, 'accuracy': 0.75},
                'test_metrics': {'f1': 0.78, 'accuracy': 0.73},
                'cv_mean': 0.79,
                'cv_std': 0.02,
                'feature_importance': None,
                'mlflow_run_id': 'run_a'
            },
            'model_b': {
                'status': 'success',
                'model': 'mock_model_b',
                'val_metrics': {'f1': 0.85, 'accuracy': 0.82},  # Better model
                'test_metrics': {'f1': 0.83, 'accuracy': 0.80},
                'cv_mean': 0.84,
                'cv_std': 0.03,
                'feature_importance': None,
                'mlflow_run_id': 'run_b'
            },
            'model_c': {
                'status': 'failed',
                'error': 'Mock error'
            }
        }
        
        best_model = agent._select_best_model(training_results)
        
        # Should select model_b as it has highest F1 score
        assert best_model['model_name'] == 'model_b'
        assert best_model['primary_metric'] == 'f1'
        assert best_model['primary_metric_value'] == 0.85
    
    def test_performance_requirements_check(self):
        """Test model performance requirements checking"""
        agent = ModelTrainingAgent()
        
        # Test classification requirements - meeting requirements
        evaluation_results = {
            'test_metrics': {
                'accuracy': 0.75,
                'f1': 0.70
            }
        }
        
        meets_req = agent._check_model_requirements(evaluation_results, 'classification')
        assert meets_req == True
        
        # Test classification requirements - not meeting requirements
        evaluation_results['test_metrics'] = {
            'accuracy': 0.65,  # Below threshold
            'f1': 0.60         # Below threshold
        }
        
        meets_req = agent._check_model_requirements(evaluation_results, 'classification')
        assert meets_req == False
        
        # Test regression requirements
        evaluation_results = {
            'test_metrics': {
                'r2': 0.60,
                'mape': 15.0
            }
        }
        
        meets_req = agent._check_model_requirements(evaluation_results, 'regression')
        assert meets_req == True
    
    def test_recommendation_generation(self):
        """Test recommendation generation based on model performance"""
        agent = ModelTrainingAgent()
        
        # Test poor performance recommendations
        evaluation_results = {
            'test_metrics': {
                'accuracy': 0.60,  # Poor
                'f1': 0.55         # Poor
            }
        }
        
        training_report = {
            'task_type': 'classification',
            'best_model': {'cv_std': 0.15}  # High variance
        }
        
        recommendations = agent._generate_model_recommendations(evaluation_results, training_report)
        
        assert len(recommendations) > 0
        assert any('accuracy' in rec.lower() for rec in recommendations)
        assert any('f1' in rec.lower() for rec in recommendations)
        assert any('variance' in rec.lower() or 'overfitting' in rec.lower() for rec in recommendations)
    
    def test_residual_analysis(self):
        """Test residual analysis for regression models"""
        agent = ModelTrainingAgent()
        
        # Create mock predictions and residuals
        predictions = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        residuals = np.array([0.1, -0.2, 0.05, 0.15, -0.1])
        
        # Test heteroscedasticity check
        hetero_result = agent._check_heteroscedasticity(predictions, residuals)
        assert 'heteroscedastic' in hetero_result
        assert 'correlation' in hetero_result
        
        # Test normality check
        normality_result = agent._check_residual_normality(residuals)
        assert 'normal' in normality_result

class TestAdvancedModelTrainer:
    
    def test_ensemble_model_creation(self):
        """Test ensemble model creation"""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import LogisticRegression
        
        # Create sample data
        X = pd.DataFrame(np.random.randn(100, 5))
        y = pd.Series(np.random.choice([0, 1], 100))
        
        # Create individual models
        models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42)
        ]
        
        # Create ensemble
        ensemble = AdvancedModelTrainer.train_ensemble_model(models, X, y)
        
        assert ensemble is not None
        assert hasattr(ensemble, 'fit')
        assert hasattr(ensemble, 'predict')
        
        # Test prediction
        predictions = ensemble.predict(X.iloc[:10])
        assert len(predictions) == 10
    
    def test_automated_feature_selection(self):
        """Test automated feature selection during training"""
        # Create data with some irrelevant features
        X, y = make_classification(n_samples=200, n_features=20, n_informative=10, 
                                 n_redundant=5, random_state=42)
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(20)])
        y_series = pd.Series(y)
        
        # Perform feature selection
        X_selected, selected_features = AdvancedModelTrainer.perform_automated_feature_selection(
            X_df, y_series, max_features=10
        )
        
        assert X_selected.shape[1] <= 10
        assert len(selected_features) == X_selected.shape[1]
        assert all(feature in X_df.columns for feature in selected_features)

class TestModelTrainingIntegration:
    
    @pytest.mark.asyncio
    async def test_end_to_end_training_pipeline(self, mock_mlflow):
        """Test complete end-to-end model training pipeline"""
        
        # Create realistic dataset
        np.random.seed(42)
        n_samples = 800
        
        # Generate features with different characteristics
        data = {
            'numeric_1': np.random.normal(50, 15, n_samples),
            'numeric_2': np.random.exponential(2, n_samples),
            'numeric_3': np.random.uniform(0, 100, n_samples),
            'encoded_cat_1': np.random.choice([0, 1, 2], n_samples),  # Already encoded
            'encoded_cat_2': np.random.choice([0, 1], n_samples),
            'interaction_feature': np.random.normal(0, 1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create realistic target with some relationship to features
        target_prob = (
            0.3 + 
            0.01 * (df['numeric_1'] - 50) / 15 +  # Normalized feature 1
            0.2 * (df['encoded_cat_1'] == 2) +     # Category effect
            0.1 * df['encoded_cat_2'] +            # Binary category
            0.1 * np.random.randn(n_samples)       # Noise
        )
        
        # Sigmoid to convert to probabilities
        target_prob = 1 / (1 + np.exp(-target_prob))
        df['target'] = np.random.binomial(1, target_prob)
        
        print(f"Target distribution: {df['target'].value_counts().to_dict()}")
        
        # Initialize agent and run training
        agent = ModelTrainingAgent()
        
        state = {
            'processed_data': df,
            'target_column': 'target',
            'project_name': 'integration_test',
            'execution_log': []
        }
        
        # Run training
        trained_state = await agent.train_models(state)
        
        # Verify training completed successfully
        assert 'trained_models' in trained_state
        assert 'best_model' in trained_state
        assert 'training_report' in trained_state
        
        training_report = trained_state['training_report']
        
        # Check that multiple models were trained
        assert training_report['models_trained'] >= 3
        assert training_report['successful_models'] >= 2
        
        # Check best model selection
        best_model = trained_state['best_model']
        assert best_model['model'] is not None
        assert 'performance' in best_model
        assert 'primary_metric_value' in best_model
        
        print(f"Best model: {best_model['model_name']}")
        print(f"Best performance: {best_model['primary_metric_value']:.3f}")
        
        # Run evaluation
        evaluated_state = await agent.evaluate_models(trained_state)
        
        # Verify evaluation completed
        assert 'evaluation_results' in evaluated_state
        assert 'model_meets_requirements' in evaluated_state
        
        evaluation_results = evaluated_state['evaluation_results']
        
        # Check evaluation components
        assert 'best_model_performance' in evaluation_results
        assert 'model_interpretation' in evaluation_results
        assert 'model_comparison' in evaluation_results
        assert 'recommendations' in evaluation_results
        
        # Check performance metrics
        performance = evaluation_results['best_model_performance']
        assert 'test_metrics' in performance
        
        test_metrics = performance['test_metrics']
        print(f"Test metrics: {test_metrics}")
        
        # Verify reasonable performance (not too strict for test data)
        assert test_metrics['accuracy'] > 0.5  # Better than random
        assert test_metrics['f1'] > 0.4        # Reasonable F1
        
        # Check model interpretation
        interpretation = evaluation_results['model_interpretation']
        assert 'model_type' in interpretation
        assert 'interpretability' in interpretation
        
        # Check recommendations
        recommendations = evaluation_results['recommendations']
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        print(f"Model recommendations: {recommendations}")
        
        # Verify state progression
        assert evaluated_state['current_step'] == 'model_evaluation'
        assert evaluated_state['next_action'] in ['deployment', 'retrain']
        
        print("✅ End-to-end model training pipeline test completed successfully")

# Utility function for running integration tests
async def run_model_training_demo():
    """Demo function to show model training capabilities"""
    
    print("🤖 MODEL TRAINING AGENT DEMO")
    print("=" * 50)
    
    # Create demo dataset
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=1000,
        n_features=15,
        n_informative=10,
        n_redundant=3,
        n_classes=2,
        n_clusters_per_class=1,
        random_state=42
    )
    
    feature_names = [f'feature_{i:02d}' for i in range(15)]
    df = pd.DataFrame(X, columns=feature_names)
    df['target'] = y
    
    print(f"📊 Demo dataset: {df.shape[0]} samples, {df.shape[1]-1} features")
    print(f"Target distribution: {pd.Series(y).value_counts().to_dict()}")
    
    # Initialize agent
    agent = ModelTrainingAgent()
    
    state = {
        'processed_data': df,
        'target_column': 'target',
        'project_name': 'model_training_demo',
        'execution_log': []
    }
    
    # Train models
    print("\n🔄 Training multiple models...")
    trained_state = await agent.train_models(state)
    
    if 'errors' in trained_state and trained_state['errors']:
        print(f"❌ Training failed: {trained_state['errors']}")
        return
    
    training_report = trained_state['training_report']
    print(f"✅ Training completed: {training_report['successful_models']}/{training_report['models_trained']} models")
    
    # Show model results
    print(f"\n🏆 Best model: {trained_state['best_model']['model_name']}")
    print(f"Primary metric: {trained_state['best_model']['primary_metric']} = {trained_state['best_model']['primary_metric_value']:.3f}")
    
    # Evaluate models
    print(f"\n🔍 Evaluating models...")
    evaluated_state = await agent.evaluate_models(trained_state)
    
    if 'errors' in evaluated_state and evaluated_state['errors']:
        print(f"❌ Evaluation failed: {evaluated_state['errors']}")
        return
    
    evaluation_results = evaluated_state['evaluation_results']
    
    # Show evaluation results
    print(f"✅ Evaluation completed")
    print(f"Model meets requirements: {'Yes' if evaluated_state['model_meets_requirements'] else 'No'}")
    
    # Show test performance
    test_metrics = evaluation_results['best_model_performance']['test_metrics']
    print(f"\n📈 Test Performance:")
    for metric, value in test_metrics.items():
        print(f"  • {metric.upper()}: {value:.3f}")
    
    # Show top features
    if 'model_interpretation' in evaluation_results:
        interpretation = evaluation_results['model_interpretation']
        if 'top_features' in interpretation:
            print(f"\n🎯 Top Features:")
            for i, (feature, importance) in enumerate(list(interpretation['top_features'].items())[:5], 1):
                print(f"  {i}. {feature}: {importance:.3f}")
    
    # Show recommendations
    recommendations = evaluation_results['recommendations']
    print(f"\n💡 Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"  {i}. {rec}")
    
    print(f"\n🎉 Model training demo completed!")

if __name__ == "__main__":
    import asyncio
    asyncio.run(run_model_training_demo())