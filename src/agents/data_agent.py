# src/agents/data_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging
from datetime import datetime
from pathlib import Path
import great_expectations as ge
from great_expectations.core import ExpectationSuite

logger = logging.getLogger(__name__)

class DataIngestionAgent:
    """Agent responsible for data ingestion and initial validation"""
    
    def __init__(self):
        self.supported_formats = ['.csv', '.xlsx', '.json', '.parquet']
        self.max_file_size_mb = 500
    
    async def process(self, state: dict) -> dict:
        """Main processing function for data ingestion"""
        logger.info(f"Starting data ingestion for: {state['data_path']}")
        
        try:
            # Load data
            data = await self._load_data(state['data_path'])
            
            # Basic info extraction
            data_info = self._extract_data_info(data)
            
            # Update state
            state.update({
                'raw_data': data,
                'data_info': data_info,
                'current_step': 'data_ingestion',
                'next_action': 'data_validation'
            })
            
            state['execution_log'].append(
                f"Data loaded successfully: {data.shape[0]} rows, {data.shape[1]} columns"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {str(e)}")
            state['errors'].append(f"Data ingestion error: {str(e)}")
            state['next_action'] = 'error'
            return state
    
    async def _load_data(self, data_path: str) -> pd.DataFrame:
        """Load data from various file formats"""
        path = Path(data_path)
        
        # Validate file exists
        if not path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        
        # Check file size
        file_size_mb = path.stat().st_size / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            raise ValueError(f"File too large: {file_size_mb:.1f}MB > {self.max_file_size_mb}MB")
        
        # Load based on file extension
        extension = path.suffix.lower()
        
        if extension == '.csv':
            # Try different encodings and separators
            for encoding in ['utf-8', 'latin-1', 'cp1252']:
                for sep in [',', ';', '\t']:
                    try:
                        data = pd.read_csv(data_path, encoding=encoding, sep=sep)
                        if data.shape[1] > 1:  # Successfully parsed multiple columns
                            return data
                    except:
                        continue
            raise ValueError("Could not parse CSV file with any encoding/separator combination")
        
        elif extension == '.xlsx':
            return pd.read_excel(data_path)
        
        elif extension == '.json':
            return pd.read_json(data_path)
        
        elif extension == '.parquet':
            return pd.read_parquet(data_path)
        
        else:
            raise ValueError(f"Unsupported file format: {extension}")
    
    def _extract_data_info(self, data: pd.DataFrame) -> dict:
        """Extract comprehensive information about the dataset"""
        return {
            'shape': data.shape,
            'columns': list(data.columns),
            'dtypes': data.dtypes.to_dict(),
            'missing_values': data.isnull().sum().to_dict(),
            'memory_usage_mb': data.memory_usage(deep=True).sum() / (1024 * 1024),
            'numeric_columns': list(data.select_dtypes(include=[np.number]).columns),
            'categorical_columns': list(data.select_dtypes(include=['object', 'category']).columns),
            'datetime_columns': list(data.select_dtypes(include=['datetime64']).columns),
            'duplicate_rows': data.duplicated().sum(),
            'sample_data': data.head(3).to_dict('records')
        }
    
    async def validate(self, state: dict) -> dict:
        """Comprehensive data validation using Great Expectations"""
        logger.info("Starting data validation")
        
        try:
            data = state['raw_data']
            target_column = state['target_column']
            
            # Create Great Expectations DataFrame
            ge_df = ge.from_pandas(data)
            
            # Basic validation checks
            validation_results = []
            
            # 1. Check if target column exists
            if target_column not in data.columns:
                validation_results.append({
                    'check': 'target_column_exists',
                    'passed': False,
                    'message': f"Target column '{target_column}' not found"
                })
            else:
                validation_results.append({
                    'check': 'target_column_exists',
                    'passed': True,
                    'message': f"Target column '{target_column}' found"
                })
            
            # 2. Check for minimum number of rows
            min_rows = 100
            has_min_rows = len(data) >= min_rows
            validation_results.append({
                'check': 'minimum_rows',
                'passed': has_min_rows,
                'message': f"Dataset has {len(data)} rows (minimum: {min_rows})"
            })
            
            # 3. Check for excessive missing values
            missing_threshold = 0.5
            high_missing_cols = []
            for col in data.columns:
                missing_pct = data[col].isnull().sum() / len(data)
                if missing_pct > missing_threshold:
                    high_missing_cols.append(col)
            
            validation_results.append({
                'check': 'missing_values',
                'passed': len(high_missing_cols) == 0,
                'message': f"Columns with >50% missing: {high_missing_cols}" if high_missing_cols else "Missing values within acceptable range"
            })
            
            # 4. Check target variable distribution
            if target_column in data.columns:
                target_validation = self._validate_target_variable(data[target_column])
                validation_results.append(target_validation)
            
            # 5. Check for data types consistency
            dtype_issues = self._check_data_types(data)
            validation_results.append({
                'check': 'data_types',
                'passed': len(dtype_issues) == 0,
                'message': f"Data type issues: {dtype_issues}" if dtype_issues else "Data types are consistent"
            })
            
            # 6. Check for duplicate rows
            duplicate_pct = data.duplicated().sum() / len(data)
            validation_results.append({
                'check': 'duplicates',
                'passed': duplicate_pct < 0.1,
                'message': f"Duplicate rows: {duplicate_pct:.1%}"
            })
            
            # Compile validation report
            passed_checks = sum(1 for result in validation_results if result['passed'])
            total_checks = len(validation_results)
            
            validation_report = {
                'is_valid': passed_checks == total_checks,
                'can_fix': passed_checks >= total_checks * 0.7,  # Can proceed if 70% checks pass
                'passed_checks': passed_checks,
                'total_checks': total_checks,
                'results': validation_results,
                'recommendations': self._generate_recommendations(validation_results)
            }
            
            state.update({
                'validation_report': validation_report,
                'current_step': 'data_validation',
                'next_action': 'proceed' if validation_report['is_valid'] else ('retry' if validation_report['can_fix'] else 'error')
            })
            
            state['execution_log'].append(
                f"Data validation completed: {passed_checks}/{total_checks} checks passed"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Data validation failed: {str(e)}")
            state['errors'].append(f"Data validation error: {str(e)}")
            state['next_action'] = 'error'
            return state
    
    def _validate_target_variable(self, target_series: pd.Series) -> dict:
        """Validate the target variable"""
        target_info = {
            'name': target_series.name,
            'dtype': str(target_series.dtype),
            'unique_values': target_series.nunique(),
            'missing_count': target_series.isnull().sum(),
            'missing_pct': target_series.isnull().sum() / len(target_series)
        }
        
        # Determine if classification or regression
        if target_series.dtype in ['object', 'category'] or target_series.nunique() < 20:
            # Classification
            task_type = 'classification'
            class_distribution = target_series.value_counts(normalize=True).to_dict()
            
            # Check for class imbalance
            min_class_pct = min(class_distribution.values())
            is_balanced = min_class_pct >= 0.1  # At least 10% for minority class
            
            validation_message = f"Classification task: {target_series.nunique()} classes, {'balanced' if is_balanced else 'imbalanced'}"
            
        else:
            # Regression
            task_type = 'regression'
            target_stats = {
                'mean': float(target_series.mean()),
                'std': float(target_series.std()),
                'min': float(target_series.min()),
                'max': float(target_series.max())
            }
            
            # Check for outliers using IQR method
            Q1 = target_series.quantile(0.25)
            Q3 = target_series.quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((target_series < (Q1 - 1.5 * IQR)) | (target_series > (Q3 + 1.5 * IQR))).sum()
            outlier_pct = outliers / len(target_series)
            
            validation_message = f"Regression task: {outlier_pct:.1%} outliers detected"
        
        return {
            'check': 'target_variable',
            'passed': target_info['missing_pct'] < 0.05,  # Less than 5% missing
            'message': validation_message,
            'details': target_info
        }
    
    def _check_data_types(self, data: pd.DataFrame) -> List[str]:
        """Check for data type inconsistencies"""
        issues = []
        
        for col in data.columns:
            if data[col].dtype == 'object':
                # Check if numeric data is stored as string
                non_null_values = data[col].dropna()
                if len(non_null_values) > 0:
                    # Try to convert to numeric
                    try:
                        pd.to_numeric(non_null_values.iloc[:min(100, len(non_null_values))])
                        issues.append(f"Column '{col}' appears numeric but stored as text")
                    except:
                        pass
        
        return issues
    
    def _generate_recommendations(self, validation_results: List[dict]) -> List[str]:
        """Generate recommendations based on validation results"""
        recommendations = []
        
        for result in validation_results:
            if not result['passed']:
                check_type = result['check']
                
                if check_type == 'target_column_exists':
                    recommendations.append("Verify target column name or provide correct column name")
                elif check_type == 'minimum_rows':
                    recommendations.append("Consider collecting more data or using data augmentation techniques")
                elif check_type == 'missing_values':
                    recommendations.append("Implement missing value imputation or consider dropping high-missing columns")
                elif check_type == 'data_types':
                    recommendations.append("Clean and convert data types before proceeding")
                elif check_type == 'duplicates':
                    recommendations.append("Remove duplicate rows to improve data quality")
        
        return recommendations

# Utility functions for data preprocessing
class DataPreprocessor:
    """Utility class for data preprocessing operations"""
    
    @staticmethod
    def auto_detect_dtypes(data: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and convert data types"""
        data_copy = data.copy()
        
        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                # Try to convert to numeric
                try:
                    data_copy[col] = pd.to_numeric(data_copy[col], errors='ignore')
                except:
                    pass
                
                # Try to convert to datetime
                if data_copy[col].dtype == 'object':
                    try:
                        data_copy[col] = pd.to_datetime(data_copy[col], errors='ignore')
                    except:
                        pass
        
        return data_copy
    
    @staticmethod
    def handle_missing_values(data: pd.DataFrame, strategy: str = 'auto') -> pd.DataFrame:
        """Handle missing values with various strategies"""
        data_copy = data.copy()
        
        if strategy == 'auto':
            for col in data_copy.columns:
                if data_copy[col].dtype in ['int64', 'float64']:
                    # Use median for numeric columns
                    data_copy[col].fillna(data_copy[col].median(), inplace=True)
                else:
                    # Use mode for categorical columns
                    mode_value = data_copy[col].mode().iloc[0] if not data_copy[col].mode().empty else 'Unknown'
                    data_copy[col].fillna(mode_value, inplace=True)
        
        return data_copy