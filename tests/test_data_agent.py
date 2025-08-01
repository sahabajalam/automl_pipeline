# tests/test_data_agent.py
import pytest
import pandas as pd
import numpy as np
from src.agents.data_agent import DataIngestionAgent, DataPreprocessor
import tempfile
import os

class TestDataIngestionAgent:
    
    @pytest.fixture
    def sample_data(self):
        """Create sample dataset for testing"""
        np.random.seed(42)
        data = {
            'age': np.random.randint(18, 65, 1000),
            'income': np.random.normal(50000, 15000, 1000),
            'education': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 1000),
            'experience_years': np.random.randint(0, 40, 1000),
            'target': np.random.choice([0, 1], 1000, p=[0.3, 0.7])
        }
        
        # Introduce some missing values
        data_df = pd.DataFrame(data)
        data_df.loc[np.random.choice(data_df.index, 50), 'income'] = np.nan
        data_df.loc[np.random.choice(data_df.index, 20), 'education'] = np.nan
        
        return data_df
    
    @pytest.fixture
    def problematic_data(self):
        """Create problematic dataset for testing validation"""
        data = {
            'col1': [1, 2, 3, 4, 5] * 20,  # 100 rows
            'col2': [np.nan] * 80 + [1, 2, 3, 4, 5] * 4,  # 80% missing
            'col3': ['A', 'B', 'C', 'D', 'E'] * 20,
            'wrong_target': [0, 1] * 50  # Wrong target column name
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def temp_csv_file(self, sample_data):
        """Create temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)
    
    @pytest.mark.asyncio
    async def test_data_loading_success(self, temp_csv_file):
        """Test successful data loading"""
        agent = DataIngestionAgent()
        
        state = {
            'data_path': temp_csv_file,
            'target_column': 'target',
            'execution_log': []
        }
        
        result = await agent.process(state)
        
        assert 'raw_data' in result
        assert isinstance(result['raw_data'], pd.DataFrame)
        assert result['raw_data'].shape[0] > 0
        assert result['current_step'] == 'data_ingestion'
        assert result['next_action'] == 'data_validation'
    
    @pytest.mark.asyncio
    async def test_data_loading_file_not_found(self):
        """Test data loading with non-existent file"""
        agent = DataIngestionAgent()
        
        state = {
            'data_path': 'non_existent_file.csv',
            'target_column': 'target',
            'execution_log': [],
            'errors': []
        }
        
        result = await agent.process(state)
        
        assert 'errors' in result
        assert len(result['errors']) > 0
        assert result['next_action'] == 'error'
    
    @pytest.mark.asyncio
    async def test_validation_success(self, sample_data):
        """Test successful data validation"""
        agent = DataIngestionAgent()
        
        state = {
            'raw_data': sample_data,
            'target_column': 'target',
            'execution_log': []
        }
        
        result = await agent.validate(state)
        
        assert 'validation_report' in result
        assert result['validation_report']['is_valid'] == True
        assert result['current_step'] == 'data_validation'
        assert result['next_action'] == 'proceed'
    
    @pytest.mark.asyncio
    async def test_validation_failure(self, problematic_data):
        """Test data validation with problematic data"""
        agent = DataIngestionAgent()
        
        state = {
            'raw_data': problematic_data,
            'target_column': 'target',  # This column doesn't exist
            'execution_log': []
        }
        
        result = await agent.validate(state)
        
        assert 'validation_report' in result
        assert result['validation_report']['is_valid'] == False
        assert len(result['validation_report']['results']) > 0
    
    def test_data_info_extraction(self, sample_data):
        """Test data information extraction"""
        agent = DataIngestionAgent()
        info = agent._extract_data_info(sample_data)
        
        assert 'shape' in info
        assert 'columns' in info
        assert 'dtypes' in info
        assert 'missing_values' in info
        assert info['shape'] == sample_data.shape
        assert set(info['columns']) == set(sample_data.columns)

class TestDataPreprocessor:
    
    def test_auto_detect_dtypes(self):
        """Test automatic data type detection"""
        data = pd.DataFrame({
            'numeric_as_string': ['1', '2', '3', '4'],
            'actual_string': ['A', 'B', 'C', 'D'],
            'datetime_string': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04']
        })
        
        processed = DataPreprocessor.auto_detect_dtypes(data)
        
        # Should convert numeric strings to numeric
        assert pd.api.types.is_numeric_dtype(processed['numeric_as_string'])
        # Should keep actual strings as strings
        assert processed['actual_string'].dtype == 'object'
        # Should convert datetime strings to datetime
        assert pd.api.types.is_datetime64_any_dtype(processed['datetime_string'])
    
    def test_handle_missing_values(self):
        """Test missing value handling"""
        data = pd.DataFrame({
            'numeric_col': [1, 2, np.nan, 4, 5],
            'string_col': ['A', 'B', np.nan, 'D', 'E']
        })
        
        processed = DataPreprocessor.handle_missing_values(data)
        
        # Should have no missing values
        assert processed.isnull().sum().sum() == 0
        # Numeric column should be filled with median
        assert processed['numeric_col'].iloc[2] == data['numeric_col'].median()

# Sample data generation script
# data/sample/generate_sample_data.py
import pandas as pd
import numpy as np
from pathlib import Path

def generate_titanic_like_dataset():
    """Generate a Titanic-like dataset for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate features
    data = {
        'passenger_id': range(1, n_samples + 1),
        'pclass': np.random.choice([1, 2, 3], n_samples, p=[0.2, 0.3, 0.5]),
        'sex': np.random.choice(['male', 'female'], n_samples, p=[0.6, 0.4]),
        'age': np.random.gamma(2, 15, n_samples),  # Age distribution
        'sibsp': np.random.poisson(0.5, n_samples),  # Siblings/spouses
        'parch': np.random.poisson(0.4, n_samples),  # Parents/children
        'ticket': [f'TICKET_{i}' for i in range(n_samples)],
        'fare': np.random.exponential(30, n_samples),
        'cabin': [f'C{i}' if np.random.random() > 0.7 else np.nan for i in range(n_samples)],
        'embarked': np.random.choice(['S', 'C', 'Q'], n_samples, p=[0.7, 0.2, 0.1])
    }
    
    df = pd.DataFrame(data)
    
    # Create target variable based on realistic survival factors
    survival_prob = (
        0.1 +  # Base survival rate
        0.4 * (df['sex'] == 'female') +  # Women more likely to survive
        0.3 * (df['pclass'] == 1) +  # First class more likely
        0.1 * (df['pclass'] == 2) +  # Second class moderate
        0.2 * (df['age'] < 16) +  # Children more likely
        -0.1 * (df['age'] > 60)  # Elderly less likely
    )
    
    # Add some randomness
    survival_prob += np.random.normal(0, 0.1, n_samples)
    survival_prob = np.clip(survival_prob, 0, 1)
    
    df['survived'] = np.random.binomial(1, survival_prob)
    
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.1 * n_samples), replace=False)
    df.loc[missing_indices[:50], 'age'] = np.nan
    df.loc[missing_indices[50:], 'embarked'] = np.nan
    
    return df

def generate_house_prices_dataset():
    """Generate a house prices dataset for regression testing"""
    np.random.seed(42)
    n_samples = 1500
    
    # Generate features
    data = {
        'id': range(1, n_samples + 1),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'sqft_living': np.random.normal(2000, 500, n_samples),
        'sqft_lot': np.random.exponential(8000, n_samples),
        'floors': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.3, 0.1, 0.4, 0.1, 0.1]),
        'waterfront': np.random.choice([0, 1], n_samples, p=[0.9, 0.1]),
        'view': np.random.randint(0, 5, n_samples),
        'condition': np.random.randint(1, 6, n_samples),
        'grade': np.random.randint(3, 13, n_samples),
        'yr_built': np.random.randint(1900, 2020, n_samples),
        'zipcode': np.random.choice([98001, 98002, 98003, 98004, 98005], n_samples),
        'lat': np.random.uniform(47.1, 47.8, n_samples),
        'long': np.random.uniform(-122.5, -121.3, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic price based on features
    price = (
        50000 +  # Base price
        df['bedrooms'] * 20000 +
        df['bathrooms'] * 15000 +
        df['sqft_living'] * 100 +
        df['sqft_lot'] * 2 +
        df['waterfront'] * 100000 +
        df['view'] * 10000 +
        df['condition'] * 5000 +
        df['grade'] * 20000 +
        (2020 - df['yr_built']) * -500  # Depreciation
    )
    
    # Add some noise
    price += np.random.normal(0, 50000, n_samples)
    price = np.maximum(price, 50000)  # Minimum price
    
    df['price'] = price.round(-3)  # Round to nearest thousand
    
    # Introduce some missing values
    missing_indices = np.random.choice(df.index, size=int(0.05 * n_samples), replace=False)
    df.loc[missing_indices[:30], 'yr_built'] = np.nan
    df.loc[missing_indices[30:], 'view'] = np.nan
    
    return df

def generate_customer_churn_dataset():
    """Generate a customer churn dataset for classification"""
    np.random.seed(42)
    n_samples = 2000
    
    # Generate features
    data = {
        'customer_id': [f'CUST_{i:05d}' for i in range(n_samples)],
        'tenure_months': np.random.exponential(20, n_samples),
        'monthly_charges': np.random.normal(65, 20, n_samples),
        'total_charges': np.random.exponential(1500, n_samples),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], 
                                        n_samples, p=[0.5, 0.3, 0.2]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'],
                                         n_samples, p=[0.4, 0.2, 0.2, 0.2]),
        'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.4, 0.4, 0.2]),
        'online_security': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'streaming_tv': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'senior_citizen': np.random.choice([0, 1], n_samples, p=[0.84, 0.16]),
        'partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
        'dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7])
    }
    
    df = pd.DataFrame(data)
    
    # Create churn probability based on realistic factors
    churn_prob = (
        0.05 +  # Base churn rate
        0.3 * (df['contract_type'] == 'Month-to-month') +
        0.1 * (df['contract_type'] == 'One year') +
        0.2 * (df['payment_method'] == 'Electronic check') +
        -0.1 * (df['partner'] == 'Yes') +
        -0.1 * (df['dependents'] == 'Yes') +
        0.1 * df['senior_citizen'] +
        -0.01 * np.minimum(df['tenure_months'], 50) +  # Longer tenure = less churn
        0.002 * np.maximum(df['monthly_charges'] - 65, 0)  # Higher charges = more churn
    )
    
    # Add randomness
    churn_prob += np.random.normal(0, 0.1, n_samples)
    churn_prob = np.clip(churn_prob, 0, 1)
    
    df['churn'] = np.random.binomial(1, churn_prob)
    
    return df

if __name__ == "__main__":
    # Create sample data directory
    sample_dir = Path("data/sample")
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate and save datasets
    print("Generating sample datasets...")
    
    # Titanic-like dataset
    titanic_data = generate_titanic_like_dataset()
    titanic_data.to_csv(sample_dir / "titanic.csv", index=False)
    print(f"Generated titanic.csv: {titanic_data.shape}")
    
    # House prices dataset
    house_data = generate_house_prices_dataset()
    house_data.to_csv(sample_dir / "house_prices.csv", index=False)
    print(f"Generated house_prices.csv: {house_data.shape}")
    
    # Customer churn dataset
    churn_data = generate_customer_churn_dataset()
    churn_data.to_csv(sample_dir / "customer_churn.csv", index=False)
    print(f"Generated customer_churn.csv: {churn_data.shape}")
    
    # Generate Excel version for testing
    titanic_data.to_excel(sample_dir / "titanic.xlsx", index=False)
    print("Generated titanic.xlsx")
    
    print("\nSample data generation complete!")
    print(f"Files saved to: {sample_dir.absolute()}")

# Run this to generate sample data
# python data/sample/generate_sample_data.py