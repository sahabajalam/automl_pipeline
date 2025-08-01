# src/config.py
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import json

@dataclass
class PathConfig:
    """Configuration for project paths"""
    PROJECT_ROOT: Path
    DATA_DIR: Path
    RAW_DATA_DIR: Path
    PROCESSED_DATA_DIR: Path
    SAMPLE_DATA_DIR: Path
    MODELS_DIR: Path
    CHECKPOINTS_DIR: Path
    DEPLOYMENTS_DIR: Path
    LOGS_DIR: Path
    NOTEBOOKS_DIR: Path
    TESTS_DIR: Path

@dataclass
class MLFlowConfig:
    """Configuration for MLflow tracking"""
    TRACKING_URI: str
    EXPERIMENT_NAME: str
    ARTIFACT_LOCATION: Optional[str]
    REGISTRY_URI: Optional[str]

@dataclass
class ModelTrainingConfig:
    """Configuration for model training"""
    TEST_SIZE: float
    VALIDATION_SIZE: float
    RANDOM_STATE: int
    CV_FOLDS: int
    MAX_TRAINING_TIME: int  # seconds
    EARLY_STOPPING_PATIENCE: int
    HYPERPARAMETER_TUNING_ITERATIONS: int

@dataclass  
class PerformanceThresholds:
    """Performance thresholds for model acceptance"""
    # Classification thresholds
    MIN_ACCURACY: float
    MIN_F1_SCORE: float
    MIN_PRECISION: float
    MIN_RECALL: float
    MIN_ROC_AUC: float
    
    # Regression thresholds
    MIN_R2_SCORE: float
    MAX_MAE: float
    MAX_RMSE: float
    MAX_MAPE: float

@dataclass
class DataValidationConfig:
    """Configuration for data validation"""
    MAX_FILE_SIZE_MB: int
    MIN_ROWS: int
    MAX_MISSING_PERCENTAGE: float
    MAX_DUPLICATE_PERCENTAGE: float
    SUPPORTED_FILE_FORMATS: List[str]

@dataclass
class FeatureEngineeringConfig:
    """Configuration for feature engineering"""
    MAX_FEATURES: int
    MIN_FEATURE_IMPORTANCE: float
    CORRELATION_THRESHOLD: float
    VARIANCE_THRESHOLD: float
    OUTLIER_METHOD: str  # 'iqr', 'zscore', 'isolation_forest'
    OUTLIER_THRESHOLD: float

@dataclass
class DeploymentConfig:
    """Configuration for model deployment"""
    DEFAULT_PORT: int
    DEFAULT_HOST: str
    WORKERS: int
    TIMEOUT: int
    MAX_REQUEST_SIZE: int
    RATE_LIMIT_PER_MINUTE: int
    ENABLE_CORS: bool
    ENABLE_DOCS: bool

class Config:
    """Central configuration manager for the ML pipeline"""
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize configuration
        
        Args:
            config_file: Optional path to JSON config file to override defaults
        """
        self._load_default_config()
        
        if config_file and os.path.exists(config_file):
            self._load_config_file(config_file)
        
        self._load_environment_variables()
        self.create_directories()
    
    def _load_default_config(self):
        """Load default configuration values"""
        
        # Project paths
        project_root = Path(__file__).parent.parent
        self.paths = PathConfig(
            PROJECT_ROOT=project_root,
            DATA_DIR=project_root / "data",
            RAW_DATA_DIR=project_root / "data" / "raw",
            PROCESSED_DATA_DIR=project_root / "data" / "processed",
            SAMPLE_DATA_DIR=project_root / "data" / "sample",
            MODELS_DIR=project_root / "models",
            CHECKPOINTS_DIR=project_root / "checkpoints",
            DEPLOYMENTS_DIR=project_root / "deployments",
            LOGS_DIR=project_root / "logs",
            NOTEBOOKS_DIR=project_root / "notebooks",
            TESTS_DIR=project_root / "tests"
        )
        
        # MLflow configuration
        self.mlflow = MLFlowConfig(
            TRACKING_URI="sqlite:///mlflow.db",
            EXPERIMENT_NAME="autonomous_ml_pipeline",
            ARTIFACT_LOCATION=None,
            REGISTRY_URI=None
        )
        
        # Model training configuration
        self.training = ModelTrainingConfig(
            TEST_SIZE=0.2,
            VALIDATION_SIZE=0.2,
            RANDOM_STATE=42,
            CV_FOLDS=3,
            MAX_TRAINING_TIME=3600,  # 1 hour
            EARLY_STOPPING_PATIENCE=10,
            HYPERPARAMETER_TUNING_ITERATIONS=20
        )
        
        # Performance thresholds
        self.thresholds = PerformanceThresholds(
            # Classification
            MIN_ACCURACY=0.7,
            MIN_F1_SCORE=0.65,
            MIN_PRECISION=0.6,
            MIN_RECALL=0.6,
            MIN_ROC_AUC=0.7,
            # Regression
            MIN_R2_SCORE=0.5,
            MAX_MAE=1000.0,
            MAX_RMSE=1000.0,
            MAX_MAPE=20.0
        )
        
        # Data validation configuration
        self.data_validation = DataValidationConfig(
            MAX_FILE_SIZE_MB=500,
            MIN_ROWS=100,
            MAX_MISSING_PERCENTAGE=50.0,
            MAX_DUPLICATE_PERCENTAGE=10.0,
            SUPPORTED_FILE_FORMATS=['.csv', '.xlsx', '.json', '.parquet']
        )
        
        # Feature engineering configuration
        self.feature_engineering = FeatureEngineeringConfig(
            MAX_FEATURES=50,
            MIN_FEATURE_IMPORTANCE=0.01,
            CORRELATION_THRESHOLD=0.9,
            VARIANCE_THRESHOLD=0.01,
            OUTLIER_METHOD='iqr',
            OUTLIER_THRESHOLD=1.5
        )
        
        # Deployment configuration
        self.deployment = DeploymentConfig(
            DEFAULT_PORT=8000,
            DEFAULT_HOST="0.0.0.0",
            WORKERS=1,
            TIMEOUT=30,
            MAX_REQUEST_SIZE=10 * 1024 * 1024,  # 10MB
            RATE_LIMIT_PER_MINUTE=1000,
            ENABLE_CORS=True,
            ENABLE_DOCS=True
        )
        
        # Additional settings
        self.logging_level = "INFO"
        self.debug_mode = False
        self.parallel_processing = True
        self.max_workers = os.cpu_count()
    
    def _load_config_file(self, config_file: str):
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
            
            # Update configurations with values from file
            for section, values in config_data.items():
                if hasattr(self, section):
                    config_obj = getattr(self, section)
                    for key, value in values.items():
                        if hasattr(config_obj, key):
                            setattr(config_obj, key, value)
        
        except Exception as e:
            print(f"Warning: Could not load config file {config_file}: {e}")
    
    def _load_environment_variables(self):
        """Load configuration from environment variables"""
        
        # MLflow settings
        if os.getenv("MLFLOW_TRACKING_URI"):
            self.mlflow.TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI")
        
        if os.getenv("MLFLOW_EXPERIMENT_NAME"):
            self.mlflow.EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")
        
        # Training settings
        if os.getenv("TEST_SIZE"):
            self.training.TEST_SIZE = float(os.getenv("TEST_SIZE"))
        
        if os.getenv("RANDOM_STATE"):
            self.training.RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
        
        if os.getenv("CV_FOLDS"):
            self.training.CV_FOLDS = int(os.getenv("CV_FOLDS"))
        
        # Performance thresholds
        if os.getenv("MIN_ACCURACY"):
            self.thresholds.MIN_ACCURACY = float(os.getenv("MIN_ACCURACY"))
        
        if os.getenv("MIN_F1_SCORE"):
            self.thresholds.MIN_F1_SCORE = float(os.getenv("MIN_F1_SCORE"))
        
        if os.getenv("MIN_R2_SCORE"):
            self.thresholds.MIN_R2_SCORE = float(os.getenv("MIN_R2_SCORE"))
        
        # Deployment settings
        if os.getenv("API_PORT"):
            self.deployment.DEFAULT_PORT = int(os.getenv("API_PORT"))
        
        if os.getenv("API_HOST"):
            self.deployment.DEFAULT_HOST = os.getenv("API_HOST")
        
        if os.getenv("API_WORKERS"):
            self.deployment.WORKERS = int(os.getenv("API_WORKERS"))
        
        # General settings
        if os.getenv("LOG_LEVEL"):
            self.logging_level = os.getenv("LOG_LEVEL")
        
        if os.getenv("DEBUG_MODE"):
            self.debug_mode = os.getenv("DEBUG_MODE").lower() == 'true'
        
        if os.getenv("MAX_WORKERS"):
            self.max_workers = int(os.getenv("MAX_WORKERS"))
    
    def create_directories(self):
        """Create necessary directories if they don't exist"""
        directories = [
            self.paths.DATA_DIR,
            self.paths.RAW_DATA_DIR,
            self.paths.PROCESSED_DATA_DIR,
            self.paths.SAMPLE_DATA_DIR,
            self.paths.MODELS_DIR,
            self.paths.CHECKPOINTS_DIR,
            self.paths.DEPLOYMENTS_DIR,
            self.paths.LOGS_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_model_config(self, task_type: str) -> Dict[str, Any]:
        """Get model-specific configuration based on task type"""
        
        base_config = {
            'random_state': self.training.RANDOM_STATE,
            'cv_folds': self.training.CV_FOLDS,
            'max_training_time': self.training.MAX_TRAINING_TIME,
            'early_stopping_patience': self.training.EARLY_STOPPING_PATIENCE,
            'hyperparameter_iterations': self.training.HYPERPARAMETER_TUNING_ITERATIONS
        }
        
        if task_type == 'classification':
            base_config.update({
                'min_accuracy': self.thresholds.MIN_ACCURACY,
                'min_f1_score': self.thresholds.MIN_F1_SCORE,
                'min_precision': self.thresholds.MIN_PRECISION,
                'min_recall': self.thresholds.MIN_RECALL,
                'min_roc_auc': self.thresholds.MIN_ROC_AUC
            })
        else:  # regression
            base_config.update({
                'min_r2_score': self.thresholds.MIN_R2_SCORE,
                'max_mae': self.thresholds.MAX_MAE,
                'max_rmse': self.thresholds.MAX_RMSE,
                'max_mape': self.thresholds.MAX_MAPE
            })
        
        return base_config
    
    def save_config(self, config_file: str):
        """Save current configuration to JSON file"""
        config_dict = {}
        
        # Convert dataclasses to dictionaries
        for attr_name in dir(self):
            if not attr_name.startswith('_'):
                attr_value = getattr(self, attr_name)
                if hasattr(attr_value, '__dict__'):
                    # Convert dataclass to dict
                    config_dict[attr_name] = {}
                    for field_name, field_value in attr_value.__dict__.items():
                        if isinstance(field_value, Path):
                            config_dict[attr_name][field_name] = str(field_value)
                        else:
                            config_dict[attr_name][field_name] = field_value
                elif not callable(attr_value):
                    config_dict[attr_name] = attr_value
        
        with open(config_file, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def validate_config(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate paths exist
        if not self.paths.PROJECT_ROOT.exists():
            issues.append(f"Project root does not exist: {self.paths.PROJECT_ROOT}")
        
        # Validate thresholds
        if self.training.TEST_SIZE <= 0 or self.training.TEST_SIZE >= 1:
            issues.append(f"Invalid test size: {self.training.TEST_SIZE}")
        
        if self.training.CV_FOLDS < 2:
            issues.append(f"CV folds must be >= 2: {self.training.CV_FOLDS}")
        
        if self.thresholds.MIN_ACCURACY < 0 or self.thresholds.MIN_ACCURACY > 1:
            issues.append(f"Invalid accuracy threshold: {self.thresholds.MIN_ACCURACY}")
        
        # Validate data validation settings
        if self.data_validation.MAX_FILE_SIZE_MB <= 0:
            issues.append(f"Invalid max file size: {self.data_validation.MAX_FILE_SIZE_MB}")
        
        if self.data_validation.MIN_ROWS <= 0:
            issues.append(f"Invalid min rows: {self.data_validation.MIN_ROWS}")
        
        return issues
    
    def __str__(self) -> str:
        """String representation of configuration"""
        return f"Config(project_root={self.paths.PROJECT_ROOT}, debug={self.debug_mode})"

# Global configuration instance
_config = None

def get_config(config_file: Optional[str] = None) -> Config:
    """Get global configuration instance (singleton pattern)"""
    global _config
    if _config is None:
        _config = Config(config_file)
    return _config

def reload_config(config_file: Optional[str] = None) -> Config:
    """Reload configuration (useful for testing)"""
    global _config
    _config = Config(config_file)
    return _config

# Example configuration file template
CONFIG_TEMPLATE = {
    "mlflow": {
        "TRACKING_URI": "sqlite:///mlflow.db",
        "EXPERIMENT_NAME": "my_ml_experiment"
    },
    "training": {
        "TEST_SIZE": 0.2,
        "RANDOM_STATE": 42,
        "CV_FOLDS": 5
    },
    "thresholds": {
        "MIN_ACCURACY": 0.8,
        "MIN_F1_SCORE": 0.75,
        "MIN_R2_SCORE": 0.6
    },
    "deployment": {
        "DEFAULT_PORT": 8080,
        "WORKERS": 2
    }
}

def create_config_template(output_file: str):
    """Create a configuration template file"""
    with open(output_file, 'w') as f:
        json.dump(CONFIG_TEMPLATE, f, indent=2)
    print(f"Configuration template created: {output_file}")