# tests/test_feature_agent.py
import pytest
import pandas as pd
import numpy as np
from src.agents.feature_agent import FeatureEngineeringAgent, AdvancedFeatureEngineer, FeatureAnalyzer
from sklearn.datasets import make_classification, make_regression

class TestFeatureEngineeringAgent:
    
    @pytest.fixture
    def sample_classification_data(self):
        """Create sample classification dataset"""
        X, y = make_classification(n_samples=1000, n_features=10, n_classes=2, random_state=42)
        
        # Convert to DataFrame with mixed data types
        feature_names = [f'numeric_{i}' for i in range(7)] + ['cat_1', 'cat_2', 'datetime_col']
        df = pd.DataFrame(X[:, :7], columns=feature_names[:7])
        
        # Add categorical columns
        df['cat_1'] = np.random.choice(['A', 'B', 'C'], size=1000)
        df['cat_2'] = np.random.choice(['X', 'Y'], size=1000)
        
        # Add datetime column
        df['datetime_col'] = pd.date_range('2020-01-01', periods=1000, freq='D')
        
        # Add target
        df['target'] = y
        
        # Introduce some missing values
        df.loc[np.random.choice(df.index, 50), 'numeric_1'] = np.nan
        df.loc[np.random.choice(df.index, 30), 'cat_1'] = np.nan
        
        return df
    
    @pytest.fixture
    def sample_regression_data(self):
        """Create sample regression dataset"""
        X, y = make_regression(n_samples=800, n_features=8, noise=0.1, random_state=42)
        
        feature_names = [f'feature_{i}' for i in range(8)]
        df = pd.DataFrame(X, columns=feature_names)
        df['target'] = y
        
        return df
    
    @pytest.fixture
    def problematic_data(self):
        """Create dataset with various data quality issues"""
        data = {
            'good_numeric': range(100),
            'high_missing': [1] * 20 + [np.nan] * 80,
            'constant_col': [5] * 100,
            'outlier_col': [1] * 95 + [1000, 2000, 3000, 4000, 5000],  # 5% outliers
            'high_cardinality': [f'cat_{i}' for i in range(100)],  # Unique categories
            'duplicate_info': [1, 2, 3, 4, 5] * 20,  # Duplicate of good_numeric pattern
            'target': np.random.choice([0, 1], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.mark.asyncio
    async def test_data_cleaning_success(self, sample_classification_data):
        """Test successful data cleaning"""
        agent = FeatureEngineeringAgent()
        
        state = {
            'raw_data': sample_classification_data,
            'target_column': 'target',
            'execution_log': []
        }
        
        result = await agent.clean_data(state)
        
        assert 'cleaned_data' in result
        assert 'cleaning_report' in result
        assert result['current_step'] == 'data_cleaning'
        assert result['next_action'] == 'feature_engineering'
        
        # Check that data was actually cleaned
        cleaned_data = result['cleaned_data']
        assert cleaned_data.shape[0] <= sample_classification_data.shape[0]  # May have removed rows
        assert 'target' in cleaned_data.columns
    
    @pytest.mark.asyncio
    async def test_feature_engineering_classification(self, sample_classification_data):
        """Test feature engineering for classification task"""
        agent = FeatureEngineeringAgent()
        
        # First clean the data
        state = {
            'raw_data': sample_classification_data,
            'target_column': 'target',
            'execution_log': []
        }
        
        cleaned_state = await agent.clean_data(state)
        engineered_state = await agent.engineer_features(cleaned_state)
        
        assert 'processed_data' in engineered_state
        assert 'feature_report' in engineered_state
        assert engineered_state['current_step'] == 'feature_engineering'
        assert engineered_state['next_action'] == 'model_training'
        
        # Check feature engineering results
        processed_data = engineered_state['processed_data']
        feature_report = engineered_state['feature_report']
        
        assert 'target' in processed_data.columns
        assert feature_report['target_type'] == 'classification'
        assert len(feature_report['feature_names']) > 0
    
    @pytest.mark.asyncio
    async def test_feature_engineering_regression(self, sample_regression_data):
        """Test feature engineering for regression task"""
        agent = FeatureEngineeringAgent()
        
        state = {
            'cleaned_data': sample_regression_data,
            'target_column': 'target',
            'execution_log': []
        }
        
        result = await agent.engineer_features(state)
        
        assert 'processed_data' in result
        assert 'feature_report' in result
        
        feature_report = result['feature_report']
        assert feature_report['target_type'] == 'regression'
    
    @pytest.mark.asyncio
    async def test_problematic_data_handling(self, problematic_data):
        """Test handling of problematic data"""
        agent = FeatureEngineeringAgent()
        
        state = {
            'raw_data': problematic_data,
            'target_column': 'target',
            'execution_log': []
        }
        
        cleaned_state = await agent.clean_data(state)
        
        # Should handle problematic columns
        cleaned_data = cleaned_state['cleaned_data']
        
        # High missing column should be dropped
        assert 'high_missing' not in cleaned_data.columns
        
        # Constant column should be dropped in feature engineering
        engineered_state = await agent.engineer_features(cleaned_state)
        processed_data = engineered_state['processed_data']
        
        # Check that processing completed without errors
        assert 'processed_data' in engineered_state
        assert engineered_state['next_action'] == 'model_training'
    
    def test_categorical_encoding(self, sample_classification_data):
        """Test categorical encoding functionality"""
        agent = FeatureEngineeringAgent()
        
        X = sample_classification_data.drop('target', axis=1)
        X_encoded, encoding_info = agent._encode_categorical_features(X)
        
        # Should have encoded categorical columns
        assert 'cat_1' in encoding_info or 'cat_2' in encoding_info
        
        # Should not have original categorical columns (if one-hot encoded)
        categorical_cols = X.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if col in encoding_info and encoding_info[col]['method'] == 'one_hot':
                assert col not in X_encoded.columns
    
    def test_missing_value_handling(self):
        """Test missing value handling"""
        agent = FeatureEngineeringAgent()
        
        # Create data with missing values
        data = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'categorical_col': ['A', 'B', np.nan, 'D', 'E'],
            'high_missing_col': [np.nan, np.nan, np.nan, np.nan, 1],  # 80% missing
            'target': [0, 1, 0, 1, 0]
        })
        
        cleaned_data, report = agent._handle_missing_values(data, 'target')
        
        # High missing column should be dropped
        assert 'high_missing_col' not in cleaned_data.columns
        
        # Other missing values should be filled
        assert cleaned_data['numeric_col'].isnull().sum() == 0
        assert cleaned_data['categorical_col'].isnull().sum() == 0
        
        # Should have generated report
        assert len(report) > 0
    
    def test_outlier_handling(self):
        """Test outlier detection and handling"""
        agent = FeatureEngineeringAgent()
        
        # Create data with outliers  
        data = pd.DataFrame({
            'normal_col': list(range(95)) + [1000, 2000, 3000, 4000, 5000],  # 5% outliers
            'no_outlier_col': list(range(100)),
            'target': [0, 1] * 50
        })
        
        report = agent._handle_outliers(data, 'target')
        
        # Should detect and report outlier handling
        assert len(report) > 0
        outlier_report = [r for r in report if 'normal_col' in r]
        assert len(outlier_report) > 0

class TestAdvancedFeatureEngineer:
    
    def test_binned_features(self):
        """Test creation of binned features"""
        data = pd.DataFrame({
            'numeric1': range(100),
            'numeric2': np.random.normal(0, 1, 100),
            'categorical': ['A'] * 100
        })
        
        result = AdvancedFeatureEngineer.create_binned_features(
            data, ['numeric1', 'numeric2'], n_bins=5
        )
        
        assert 'numeric1_binned' in result.columns
        assert 'numeric2_binned' in result.columns
        assert result['numeric1_binned'].nunique() <= 5
    
    def test_ratio_features(self):
        """Test creation of ratio features"""
        data = pd.DataFrame({
            'num1': [1, 2, 3, 4, 5],
            'num2': [2, 4, 6, 8, 10],
            'num3': [1, 1, 1, 1, 1]  # This will create ratios
        })
        
        result = AdvancedFeatureEngineer.create_ratio_features(
            data, ['num1', 'num2', 'num3']
        )
        
        # Should create ratio columns
        ratio_cols = [col for col in result.columns if '_div_' in col]
        assert len(ratio_cols) > 0
    
    def test_cyclical_features(self):
        """Test creation of cyclical datetime features"""
        data = pd.DataFrame({
            'datetime_col': pd.date_range('2020-01-01', periods=100, freq='D'),
            'other_col': range(100)
        })
        
        result = AdvancedFeatureEngineer.detect_and_create_cyclical_features(
            data, ['datetime_col']
        )
        
        # Should create sin/cos features for month and day of week
        cyclical_cols = [col for col in result.columns if '_sin' in col or '_cos' in col]
        assert len(cyclical_cols) >= 4  # month_sin, month_cos, dow_sin, dow_cos

class TestFeatureAnalyzer:
    
    def test_feature_importance_calculation(self):
        """Test feature importance calculation"""
        X, y = make_classification(n_samples=200, n_features=10, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(10)])
        y_series = pd.Series(y)
        
        # Test mutual info method
        importance_df = FeatureAnalyzer.calculate_feature_importance(
            X_df, y_series, method='mutual_info'
        )
        
        assert len(importance_df) == 10
        assert 'mutual_info' in importance_df.columns
        assert all(importance_df['mutual_info'] >= 0)
        
        # Test all methods
        importance_df_all = FeatureAnalyzer.calculate_feature_importance(
            X_df, y_series, method='all'
        )
        
        assert 'mutual_info' in importance_df_all.columns
        assert 'random_forest' in importance_df_all.columns
        assert 'avg_rank' in importance_df_all.columns
    
    def test_correlation_analysis(self):
        """Test correlation analysis"""
        # Create data with some highly correlated features
        X = np.random.randn(100, 5)
        X[:, 4] = X[:, 0] + 0.1 * np.random.randn(100)  # Make col 4 highly correlated with col 0
        
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
        
        correlation_analysis = FeatureAnalyzer.analyze_feature_correlations(X_df, threshold=0.8)
        
        assert 'high_correlation_pairs' in correlation_analysis
        assert 'suggested_drops' in correlation_analysis
        assert 'correlation_matrix' in correlation_analysis
        
        # Should detect the high correlation between feature_0 and feature_4
        high_corr_pairs = correlation_analysis['high_correlation_pairs']
        assert len(high_corr_pairs) > 0
    
    def test_feature_report_generation(self):
        """Test comprehensive feature report generation"""
        X, y = make_classification(n_samples=200, n_features=8, random_state=42)
        X_df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        y_series = pd.Series(y)
        
        # Mock feature engineering info
        feature_engineering_info = {
            'categorical_encoding': {},
            'scaling_applied': {'method': 'standard'},
            'feature_selection': {'n_features_selected': 8}
        }
        
        report = FeatureAnalyzer.generate_feature_report(
            X_df, y_series, feature_engineering_info
        )
        
        assert 'basic_statistics' in report
        assert 'feature_types' in report
        assert 'top_features' in report
        assert 'feature_importance' in report
        assert 'correlation_analysis' in report
        assert 'engineering_applied' in report
        
        # Check basic statistics
        basic_stats = report['basic_statistics']
        assert basic_stats['n_features'] == 8
        assert basic_stats['n_samples'] == 200

# Integration test
class TestFeatureEngineeringIntegration:
    
    @pytest.mark.asyncio
    async def test_end_to_end_pipeline(self):
        """Test complete feature engineering pipeline"""
        # Create complex dataset
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'numeric1': np.random.normal(100, 15, n_samples),
            'numeric2': np.random.exponential(2, n_samples),
            'categorical1': np.random.choice(['A', 'B', 'C'], n_samples),
            'categorical2': np.random.choice(['X', 'Y'], n_samples, p=[0.7, 0.3]),
            'high_cardinality': [f'cat_{i}' for i in np.random.randint(0, 50, n_samples)],
            'datetime_col': pd.date_range('2020-01-01', periods=n_samples, freq='H'),
            'constant_col': [42] * n_samples,
            'high_missing': [1] * 100 + [np.nan] * 400
        }
        
        df = pd.DataFrame(data)
        
        # Create target variable
        df['target'] = (
            (df['numeric1'] > 100).astype(int) + 
            (df['categorical1'] == 'A').astype(int) + 
            np.random.binomial(1, 0.3, n_samples)
        ) % 2
        
        # Run complete pipeline
        agent = FeatureEngineeringAgent()
        
        state = {
            'raw_data': df,
            'target_column': 'target',
            'execution_log': []
        }
        
        # Clean data
        cleaned_state = await agent.clean_data(state)
        assert 'cleaned_data' in cleaned_state
        
        # Engineer features
        engineered_state = await agent.engineer_features(cleaned_state)
        assert 'processed_data' in engineered_state
        
        # Check final results
        processed_data = engineered_state['processed_data']
        feature_report = engineered_state['feature_report']
        
        # Should have target column
        assert 'target' in processed_data.columns
        
        # Should have removed problematic columns
        assert 'constant_col' not in processed_data.columns
        assert 'high_missing' not in processed_data.columns
        
        # Should have processed categorical variables
        original_cats = ['categorical1', 'categorical2', 'high_cardinality']
        for cat in original_cats:
            if cat in feature_report['categorical_encoding']:
                # Original categorical column should be transformed
                if feature_report['categorical_encoding'][cat]['method'] == 'one_hot':
                    assert cat not in processed_data.columns
        
        # Should have created datetime features
        datetime_features = [col for col in processed_data.columns if 'datetime_col' in col]
        assert len(datetime_features) > 0
        
        # Should have scaled numeric features
        assert feature_report['scaling_applied']['method'] == 'standard'
        
        # Should have reasonable number of final features
        assert len(processed_data.columns) > 5  # At least some features remain
        assert len(processed_data.columns) < 100  # Not too many features
        
        print(f"Final processed data shape: {processed_data.shape}")
        print(f"Feature engineering steps: {feature_report['engineering_steps']}")
        
        # Verify no data leakage - all transformations should be fit on training data only
        # (This is a basic check - in real implementation, we'd split train/test first)
        assert not processed_data.isnull().any().any()  # No missing values should remain

if __name__ == "__main__":
    # Run basic tests
    import asyncio
    
    async def run_basic_test():
        # Create simple test data
        data = pd.DataFrame({
            'feature1': range(100),
            'feature2': np.random.normal(0, 1, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'target': np.random.choice([0, 1], 100)
        })
        
        agent = FeatureEngineeringAgent()
        
        state = {
            'raw_data': data,
            'target_column': 'target',
            'execution_log': []
        }
        
        # Test cleaning
        cleaned_state = await agent.clean_data(state)
        print("✅ Data cleaning passed")
        
        # Test feature engineering
        engineered_state = await agent.engineer_features(cleaned_state)
        print("✅ Feature engineering passed")
        
        print(f"Original features: {data.shape[1]-1}")
        print(f"Final features: {engineered_state['processed_data'].shape[1]-1}")
        print(f"Engineering steps: {engineered_state['feature_report']['engineering_steps']}")
    
    asyncio.run(run_basic_test())