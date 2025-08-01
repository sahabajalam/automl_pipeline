# src/agents/feature_agent.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import SelectKBest, f_classif, f_regression, mutual_info_classif
from sklearn.decomposition import PCA
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FeatureEngineeringAgent:
    """Agent responsible for data cleaning and feature engineering"""
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_selectors = {}
        self.feature_metadata = {}
    
    async def clean_data(self, state: dict) -> dict:
        """Clean the raw data"""
        logger.info("Starting data cleaning")
        
        try:
            data = state['raw_data'].copy()
            target_column = state['target_column']
            
            # Store original shape
            original_shape = data.shape
            
            cleaning_steps = []
            
            # 1. Remove completely empty rows and columns
            initial_rows = len(data)
            data = data.dropna(how='all')
            dropped_rows = initial_rows - len(data)
            if dropped_rows > 0:
                cleaning_steps.append(f"Removed {dropped_rows} completely empty rows")
            
            # Drop columns that are all NaN
            initial_cols = len(data.columns)
            data = data.dropna(axis=1, how='all')
            dropped_cols = initial_cols - len(data.columns)
            if dropped_cols > 0:
                cleaning_steps.append(f"Removed {dropped_cols} completely empty columns")
            
            # 2. Handle duplicate rows
            initial_rows = len(data)
            data = data.drop_duplicates()
            dropped_duplicates = initial_rows - len(data)
            if dropped_duplicates > 0:
                cleaning_steps.append(f"Removed {dropped_duplicates} duplicate rows")
            
            # 3. Auto-detect and fix data types
            data = self._auto_fix_dtypes(data)
            cleaning_steps.append("Auto-corrected data types")
            
            # 4. Handle missing values intelligently
            data, missing_report = self._handle_missing_values(data, target_column)
            cleaning_steps.extend(missing_report)
            
            # 5. Remove constant columns (no variance)
            constant_cols = []
            for col in data.columns:
                if col != target_column and data[col].nunique() <= 1:
                    constant_cols.append(col)
            
            if constant_cols:
                data = data.drop(columns=constant_cols)
                cleaning_steps.append(f"Removed {len(constant_cols)} constant columns: {constant_cols}")
            
            # 6. Handle outliers in numeric columns
            outlier_report = self._handle_outliers(data, target_column)
            cleaning_steps.extend(outlier_report)
            
            # Create cleaning report
            cleaning_report = {
                'original_shape': original_shape,
                'cleaned_shape': data.shape,
                'steps_performed': cleaning_steps,
                'rows_removed': original_shape[0] - data.shape[0],
                'columns_removed': original_shape[1] - data.shape[1]
            }
            
            state.update({
                'cleaned_data': data,
                'cleaning_report': cleaning_report,
                'current_step': 'data_cleaning',
                'next_action': 'feature_engineering'
            })
            
            state['execution_log'].append(
                f"Data cleaning completed: {data.shape[0]} rows, {data.shape[1]} columns remaining"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Data cleaning failed: {str(e)}")
            state['errors'].append(f"Data cleaning error: {str(e)}")
            state['next_action'] = 'error'
            return state
    
    async def engineer_features(self, state: dict) -> dict:
        """Perform automated feature engineering"""
        logger.info("Starting feature engineering")
        
        try:
            data = state.get('cleaned_data', state['raw_data']).copy()
            target_column = state['target_column']
            
            if target_column not in data.columns:
                raise ValueError(f"Target column '{target_column}' not found in data")
            
            # Separate features and target
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            # Store original feature info
            original_features = list(X.columns)
            
            engineering_steps = []
            
            # 1. Encode categorical variables
            X, categorical_encoding_info = self._encode_categorical_features(X)
            if categorical_encoding_info:
                engineering_steps.append(f"Encoded categorical features: {list(categorical_encoding_info.keys())}")
            
            # 2. Create datetime features
            X, datetime_features = self._create_datetime_features(X)
            if datetime_features:
                engineering_steps.append(f"Created datetime features: {datetime_features}")
            
            # 3. Create interaction features (for small datasets)
            if X.shape[1] <= 20 and X.shape[0] >= 1000:
                X, interaction_features = self._create_interaction_features(X, max_interactions=10)
                if interaction_features:
                    engineering_steps.append(f"Created {len(interaction_features)} interaction features")
            
            # 4. Create polynomial features for numeric columns (if dataset is small enough)
            if X.shape[1] <= 10:
                X, poly_features = self._create_polynomial_features(X, degree=2)
                if poly_features:
                    engineering_steps.append(f"Created {len(poly_features)} polynomial features")
            
            # 5. Scale numeric features
            X, scaling_info = self._scale_numeric_features(X)
            if scaling_info:
                engineering_steps.append("Scaled numeric features")
            
            # 6. Feature selection
            X, selection_info = self._select_features(X, y, max_features=min(50, X.shape[1]))
            if selection_info:
                engineering_steps.append(f"Selected {selection_info['n_features_selected']} best features")
            
            # 7. Handle high cardinality categorical features
            X = self._handle_high_cardinality_categories(X)
            
            # Combine features and target
            processed_data = X.copy()
            processed_data[target_column] = y
            
            # Create feature engineering report
            feature_report = {
                'original_features': len(original_features),
                'final_features': len(X.columns),
                'feature_names': list(X.columns),
                'engineering_steps': engineering_steps,
                'categorical_encoding': categorical_encoding_info,
                'scaling_applied': scaling_info,
                'feature_selection': selection_info,
                'target_type': self._determine_target_type(y)
            }
            
            state.update({
                'processed_data': processed_data,
                'feature_report': feature_report,
                'current_step': 'feature_engineering',
                'next_action': 'model_training'
            })
            
            state['execution_log'].append(
                f"Feature engineering completed: {len(original_features)} -> {len(X.columns)} features"
            )
            
            return state
            
        except Exception as e:
            logger.error(f"Feature engineering failed: {str(e)}")
            state['errors'].append(f"Feature engineering error: {str(e)}")
            state['next_action'] = 'error'
            return state
    
    def _auto_fix_dtypes(self, data: pd.DataFrame) -> pd.DataFrame:
        """Automatically detect and fix data types"""
        data_copy = data.copy()
        
        for col in data_copy.columns:
            if data_copy[col].dtype == 'object':
                # Try to convert to numeric first
                numeric_converted = pd.to_numeric(data_copy[col], errors='coerce')
                if not numeric_converted.isnull().all():
                    # If most values can be converted to numeric, do it
                    non_null_original = data_copy[col].dropna()
                    non_null_converted = numeric_converted.dropna()
                    
                    if len(non_null_converted) >= 0.8 * len(non_null_original):
                        data_copy[col] = numeric_converted
                        continue
                
                # Try to convert to datetime
                try:
                    datetime_converted = pd.to_datetime(data_copy[col], errors='coerce')
                    if not datetime_converted.isnull().all():
                        non_null_original = data_copy[col].dropna()
                        non_null_converted = datetime_converted.dropna()
                        
                        if len(non_null_converted) >= 0.8 * len(non_null_original):
                            data_copy[col] = datetime_converted
                            continue
                except:
                    pass
        
        return data_copy
    
    def _handle_missing_values(self, data: pd.DataFrame, target_column: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values with intelligent strategies"""
        data_copy = data.copy()
        report = []
        
        for col in data_copy.columns:
            missing_pct = data_copy[col].isnull().sum() / len(data_copy)
            
            if missing_pct == 0:
                continue
            
            if missing_pct > 0.5:
                # Drop columns with too many missing values
                if col != target_column:
                    data_copy = data_copy.drop(columns=[col])
                    report.append(f"Dropped column '{col}' (>{missing_pct:.1%} missing)")
                continue
            
            # Fill missing values based on data type and distribution
            if data_copy[col].dtype in ['int64', 'float64']:
                # For numeric columns, use median if skewed, mean if normal
                skewness = abs(data_copy[col].skew())
                if skewness > 1:
                    fill_value = data_copy[col].median()
                    strategy = 'median'
                else:
                    fill_value = data_copy[col].mean()
                    strategy = 'mean'
                
                data_copy[col].fillna(fill_value, inplace=True)
                report.append(f"Filled '{col}' missing values with {strategy}")
                
            elif data_copy[col].dtype == 'object':
                # For categorical columns, use mode or create 'Unknown' category
                mode_values = data_copy[col].mode()
                if len(mode_values) > 0 and mode_values.iloc[0] is not None:
                    fill_value = mode_values.iloc[0]
                    data_copy[col].fillna(fill_value, inplace=True)
                    report.append(f"Filled '{col}' missing values with mode: '{fill_value}'")
                else:
                    data_copy[col].fillna('Unknown', inplace=True)
                    report.append(f"Filled '{col}' missing values with 'Unknown'")
            
            elif pd.api.types.is_datetime64_any_dtype(data_copy[col]):
                # For datetime columns, use median date
                median_date = data_copy[col].median()
                data_copy[col].fillna(median_date, inplace=True)
                report.append(f"Filled '{col}' missing values with median date")
        
        return data_copy, report
    
    def _handle_outliers(self, data: pd.DataFrame, target_column: str) -> List[str]:
        """Handle outliers in numeric columns"""
        report = []
        
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]
        
        for col in numeric_columns:
            # Use IQR method to detect outliers
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            if IQR == 0:  # No variance in data
                continue
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound)).sum()
            outlier_pct = outliers / len(data)
            
            if outlier_pct > 0.1:  # More than 10% outliers
                # Cap outliers instead of removing them
                data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
                report.append(f"Capped outliers in '{col}': {outliers} values ({outlier_pct:.1%})")
        
        return report
    
    def _encode_categorical_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Encode categorical features"""
        X_copy = X.copy()
        encoding_info = {}
        
        categorical_columns = X_copy.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            unique_values = X_copy[col].nunique()
            
            if unique_values <= 10:
                # Use one-hot encoding for low cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(X_copy[[col]])
                
                # Create column names
                feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_copy.index)
                
                # Replace original column
                X_copy = X_copy.drop(columns=[col])
                X_copy = pd.concat([X_copy, encoded_df], axis=1)
                
                self.encoders[col] = encoder
                encoding_info[col] = {'method': 'one_hot', 'categories': list(encoder.categories_[0])}
                
            else:
                # Use label encoding for high cardinality
                encoder = LabelEncoder()
                X_copy[col] = encoder.fit_transform(X_copy[col].astype(str))
                
                self.encoders[col] = encoder
                encoding_info[col] = {'method': 'label', 'classes': list(encoder.classes_)}
        
        return X_copy, scaling_info
    
    def _select_features(self, X: pd.DataFrame, y: pd.Series, max_features: int = 50) -> Tuple[pd.DataFrame, Dict]:
        """Select best features using statistical tests"""
        if X.shape[1] <= max_features:
            return X, {'n_features_selected': X.shape[1], 'method': 'all_features_kept'}
        
        # Determine if classification or regression
        task_type = self._determine_target_type(y)
        
        if task_type == 'classification':
            # Use mutual information for classification
            selector = SelectKBest(score_func=mutual_info_classif, k=max_features)
        else:
            # Use F-test for regression
            selector = SelectKBest(score_func=f_regression, k=max_features)
        
        X_selected = selector.fit_transform(X, y)
        selected_features = X.columns[selector.get_support()].tolist()
        
        X_selected_df = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
        
        self.feature_selectors['best_features'] = selector
        
        selection_info = {
            'method': 'univariate_selection',
            'n_features_selected': len(selected_features),
            'selected_features': selected_features,
            'feature_scores': selector.scores_.tolist(),
            'task_type': task_type
        }
        
        return X_selected_df, selection_info
    
    def _handle_high_cardinality_categories(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle high cardinality categorical features"""
        X_copy = X.copy()
        
        # This is mostly handled in encoding, but we can add frequency encoding here
        for col in X_copy.columns:
            if X_copy[col].dtype == 'object' and X_copy[col].nunique() > 50:
                # Convert to frequency encoding
                freq_map = X_copy[col].value_counts().to_dict()
                X_copy[col] = X_copy[col].map(freq_map)
        
        return X_copy
    
    def _determine_target_type(self, y: pd.Series) -> str:
        """Determine if target is for classification or regression"""
        if y.dtype in ['object', 'category'] or y.nunique() < 20:
            return 'classification'
        else:
            return 'regression'
    
    def transform_new_data(self, new_data: pd.DataFrame) -> pd.DataFrame:
        """Transform new data using fitted preprocessors"""
        X_new = new_data.copy()
        
        # Apply same transformations as training data
        # 1. Encode categorical features
        for col, encoder in self.encoders.items():
            if col in X_new.columns:
                if hasattr(encoder, 'categories_'):  # OneHotEncoder
                    encoded = encoder.transform(X_new[[col]])
                    feature_names = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                    encoded_df = pd.DataFrame(encoded, columns=feature_names, index=X_new.index)
                    X_new = X_new.drop(columns=[col])
                    X_new = pd.concat([X_new, encoded_df], axis=1)
                else:  # LabelEncoder
                    X_new[col] = encoder.transform(X_new[col].astype(str))
        
        # 2. Scale numeric features
        if 'standard_scaler' in self.scalers:
            scaler = self.scalers['standard_scaler']
            numeric_columns = X_new.select_dtypes(include=[np.number]).columns
            if len(numeric_columns) > 0:
                X_new[numeric_columns] = scaler.transform(X_new[numeric_columns])
        
        # 3. Select features
        if 'best_features' in self.feature_selectors:
            selector = self.feature_selectors['best_features']
            X_new = X_new.iloc[:, selector.get_support()]
        
        return X_new


# Utility class for advanced feature engineering
class AdvancedFeatureEngineer:
    """Advanced feature engineering techniques"""
    
    @staticmethod
    def create_binned_features(data: pd.DataFrame, numeric_cols: List[str], n_bins: int = 5) -> pd.DataFrame:
        """Create binned versions of numeric features"""
        data_copy = data.copy()
        
        for col in numeric_cols:
            if col in data_copy.columns:
                try:
                    binned_col = f"{col}_binned"
                    data_copy[binned_col] = pd.cut(data_copy[col], bins=n_bins, labels=False)
                except:
                    continue
        
        return data_copy
    
    @staticmethod
    def create_ratio_features(data: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
        """Create ratio features between numeric columns"""
        data_copy = data.copy()
        
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                if col1 in data_copy.columns and col2 in data_copy.columns:
                    # Avoid division by zero
                    mask = data_copy[col2] != 0
                    if mask.any():
                        ratio_col = f"{col1}_div_{col2}"
                        data_copy[ratio_col] = 0.0
                        data_copy.loc[mask, ratio_col] = data_copy.loc[mask, col1] / data_copy.loc[mask, col2]
        
        return data_copy
    
    @staticmethod
    def create_aggregated_features(data: pd.DataFrame, group_cols: List[str], agg_cols: List[str]) -> pd.DataFrame:
        """Create aggregated features by grouping"""
        data_copy = data.copy()
        
        for group_col in group_cols:
            if group_col in data_copy.columns:
                for agg_col in agg_cols:
                    if agg_col in data_copy.columns and agg_col != group_col:
                        # Create mean aggregation
                        agg_mean = data_copy.groupby(group_col)[agg_col].transform('mean')
                        data_copy[f"{agg_col}_mean_by_{group_col}"] = agg_mean
                        
                        # Create count aggregation
                        agg_count = data_copy.groupby(group_col)[agg_col].transform('count')
                        data_copy[f"{group_col}_count"] = agg_count
        
        return data_copy
    
    @staticmethod
    def detect_and_create_cyclical_features(data: pd.DataFrame, datetime_cols: List[str]) -> pd.DataFrame:
        """Create cyclical features for datetime columns"""
        data_copy = data.copy()
        
        for col in datetime_cols:
            if col in data_copy.columns and pd.api.types.is_datetime64_any_dtype(data_copy[col]):
                # Month cyclical features
                data_copy[f"{col}_month_sin"] = np.sin(2 * np.pi * data_copy[col].dt.month / 12)
                data_copy[f"{col}_month_cos"] = np.cos(2 * np.pi * data_copy[col].dt.month / 12)
                
                # Day of week cyclical features
                data_copy[f"{col}_dow_sin"] = np.sin(2 * np.pi * data_copy[col].dt.dayofweek / 7)
                data_copy[f"{col}_dow_cos"] = np.cos(2 * np.pi * data_copy[col].dt.dayofweek / 7)
                
                # Hour cyclical features (if time info available)
                if data_copy[col].dt.hour.nunique() > 1:
                    data_copy[f"{col}_hour_sin"] = np.sin(2 * np.pi * data_copy[col].dt.hour / 24)
                    data_copy[f"{col}_hour_cos"] = np.cos(2 * np.pi * data_copy[col].dt.hour / 24)
        
        return data_copy


# Feature importance and analysis utilities
class FeatureAnalyzer:
    """Analyze and rank features"""
    
    @staticmethod
    def calculate_feature_importance(X: pd.DataFrame, y: pd.Series, method: str = 'mutual_info') -> pd.DataFrame:
        """Calculate feature importance using various methods"""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
        
        # Determine task type
        task_type = 'classification' if y.nunique() < 20 or y.dtype == 'object' else 'regression'
        
        importance_scores = {}
        
        if method == 'mutual_info' or method == 'all':
            if task_type == 'classification':
                mi_scores = mutual_info_classif(X, y)
            else:
                mi_scores = mutual_info_regression(X, y)
            importance_scores['mutual_info'] = mi_scores
        
        if method == 'random_forest' or method == 'all':
            if task_type == 'classification':
                rf = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                rf = RandomForestRegressor(n_estimators=100, random_state=42)
            
            rf.fit(X, y)
            importance_scores['random_forest'] = rf.feature_importances_
        
        # Create DataFrame with results
        results = pd.DataFrame(index=X.columns)
        for score_type, scores in importance_scores.items():
            results[score_type] = scores
        
        # Add average ranking if multiple methods
        if len(importance_scores) > 1:
            # Rank each method (higher score = lower rank number)
            ranks = results.rank(ascending=False)
            results['avg_rank'] = ranks.mean(axis=1)
            results = results.sort_values('avg_rank')
        
        return results
    
    @staticmethod
    def analyze_feature_correlations(X: pd.DataFrame, threshold: float = 0.9) -> Dict:
        """Analyze feature correlations and identify highly correlated pairs"""
        correlation_matrix = X.corr().abs()
        
        # Find highly correlated pairs
        high_corr_pairs = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                corr_value = correlation_matrix.iloc[i, j]
                if corr_value > threshold:
                    high_corr_pairs.append({
                        'feature1': correlation_matrix.columns[i],
                        'feature2': correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        # Suggest features to drop
        features_to_drop = set()
        for pair in high_corr_pairs:
            # Keep the first feature, suggest dropping the second
            features_to_drop.add(pair['feature2'])
        
        return {
            'high_correlation_pairs': high_corr_pairs,
            'suggested_drops': list(features_to_drop),
            'correlation_matrix': correlation_matrix
        }
    
    @staticmethod
    def generate_feature_report(X: pd.DataFrame, y: pd.Series, 
                              feature_engineering_info: Dict) -> Dict:
        """Generate comprehensive feature analysis report"""
        
        # Basic statistics
        basic_stats = {
            'n_features': len(X.columns),
            'n_samples': len(X),
            'missing_values': X.isnull().sum().sum(),
            'memory_usage_mb': X.memory_usage(deep=True).sum() / (1024 * 1024)
        }
        
        # Feature types
        feature_types = {
            'numeric': len(X.select_dtypes(include=[np.number]).columns),
            'categorical': len(X.select_dtypes(include=['object', 'category']).columns),
            'datetime': len(X.select_dtypes(include=['datetime64']).columns)
        }
        
        # Feature importance
        importance_df = FeatureAnalyzer.calculate_feature_importance(X, y)
        top_features = importance_df.head(10).index.tolist()
        
        # Correlation analysis
        correlation_analysis = FeatureAnalyzer.analyze_feature_correlations(X)
        
        return {
            'basic_statistics': basic_stats,
            'feature_types': feature_types,
            'top_features': top_features,
            'feature_importance': importance_df.to_dict(),
            'correlation_analysis': correlation_analysis,
            'engineering_applied': feature_engineering_info
        }copy, encoding_info
    
    def _create_datetime_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Create features from datetime columns"""
        X_copy = X.copy()
        datetime_features = []
        
        datetime_columns = X_copy.select_dtypes(include=['datetime64']).columns
        
        for col in datetime_columns:
            # Extract common datetime features
            X_copy[f"{col}_year"] = X_copy[col].dt.year
            X_copy[f"{col}_month"] = X_copy[col].dt.month
            X_copy[f"{col}_day"] = X_copy[col].dt.day
            X_copy[f"{col}_dayofweek"] = X_copy[col].dt.dayofweek
            X_copy[f"{col}_quarter"] = X_copy[col].dt.quarter
            X_copy[f"{col}_is_weekend"] = (X_copy[col].dt.dayofweek >= 5).astype(int)
            
            datetime_features.extend([
                f"{col}_year", f"{col}_month", f"{col}_day",
                f"{col}_dayofweek", f"{col}_quarter", f"{col}_is_weekend"
            ])
            
            # Drop original datetime column
            X_copy = X_copy.drop(columns=[col])
        
        return X_copy, datetime_features
    
    def _create_interaction_features(self, X: pd.DataFrame, max_interactions: int = 10) -> Tuple[pd.DataFrame, List[str]]:
        """Create interaction features between numeric columns"""
        X_copy = X.copy()
        interaction_features = []
        
        numeric_columns = X_copy.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) < 2:
            return X_copy, interaction_features
        
        # Create interactions between most correlated features
        correlations = X_copy[numeric_columns].corr()
        
        # Get top correlated pairs
        pairs = []
        for i in range(len(numeric_columns)):
            for j in range(i+1, len(numeric_columns)):
                col1, col2 = numeric_columns[i], numeric_columns[j]
                corr_value = abs(correlations.iloc[i, j])
                pairs.append((col1, col2, corr_value))
        
        # Sort by correlation and take top pairs
        pairs.sort(key=lambda x: x[2], reverse=True)
        top_pairs = pairs[:max_interactions]
        
        for col1, col2, corr in top_pairs:
            interaction_name = f"{col1}_x_{col2}"
            X_copy[interaction_name] = X_copy[col1] * X_copy[col2]
            interaction_features.append(interaction_name)
        
        return X_copy, interaction_features
    
    def _create_polynomial_features(self, X: pd.DataFrame, degree: int = 2) -> Tuple[pd.DataFrame, List[str]]:
        """Create polynomial features for numeric columns"""
        X_copy = X.copy()
        poly_features = []
        
        numeric_columns = X_copy.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if degree >= 2:
                poly_name = f"{col}_squared"
                X_copy[poly_name] = X_copy[col] ** 2
                poly_features.append(poly_name)
            
            if degree >= 3:
                poly_name = f"{col}_cubed"
                X_copy[poly_name] = X_copy[col] ** 3
                poly_features.append(poly_name)
        
        return X_copy, poly_features
    
    def _scale_numeric_features(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Scale numeric features"""
        X_copy = X.copy()
        scaling_info = {}
        
        numeric_columns = X_copy.select_dtypes(include=[np.number]).columns
        
        if len(numeric_columns) > 0:
            scaler = StandardScaler()
            X_copy[numeric_columns] = scaler.fit_transform(X_copy[numeric_columns])
            
            self.scalers['standard_scaler'] = scaler
            scaling_info = {
                'method': 'standard',
                'columns': list(numeric_columns),
                'mean': scaler.mean_.tolist(),
                'scale': scaler.scale_.tolist()
            }
        
        return X_copy, scaling_info