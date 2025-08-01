# Autonomous ML Pipeline

A comprehensive, automated machine learning pipeline that handles the entire ML workflow from data ingestion to model deployment.

## Features

- 🔄 **Autonomous Workflow**: Fully automated ML pipeline using LangGraph
- 📊 **Data Processing**: Intelligent data cleaning, validation, and feature engineering
- 🤖 **Multi-Model Training**: Trains and compares multiple ML models automatically
- 🚀 **Auto Deployment**: Generates production-ready FastAPI applications with Docker
- 📈 **Monitoring**: Built-in model monitoring and observability
- 🔍 **Quality Assurance**: Comprehensive data validation and model evaluation

## Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd automl_pipeline

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Pipeline

#### Command Line Interface

```bash
# Basic usage
python main.py --data-path data/sample/titanic.csv --target-column Survived

# With custom project name
python main.py --data-path data/sample/titanic.csv --target-column Survived --project-name my_ml_project

# With custom configuration
python main.py --data-path data/sample/titanic.csv --target-column Survived --config config.json
```

#### Python API

```python
import asyncio
from src.pipeline import AutonomousMLPipeline

async def run_ml_pipeline():
    pipeline = AutonomousMLPipeline()
    
    result = await pipeline.run_pipeline(
        data_path="data/sample/titanic.csv",
        target_column="Survived",
        project_name="titanic_survival_prediction"
    )
    
    print("Pipeline Result:", result)

# Run the pipeline
asyncio.run(run_ml_pipeline())
```

#### REST API

```bash
# Start the API server
python src/api/main.py

# Run pipeline via API
curl -X POST "http://localhost:8000/pipeline/run" \
     -H "Content-Type: application/json" \
     -d '{"data_path": "data/sample/titanic.csv", "target_column": "Survived"}'

# Check status
curl "http://localhost:8000/pipeline/status/your_project_name"
```

### 3. Test the System

```bash
# Run the test script
python test_pipeline.py
```

## Project Structure

```
automl_pipeline/
├── src/
│   ├── agents/           # Pipeline agents
│   │   ├── data_agent.py      # Data ingestion & validation
│   │   ├── feature_agent.py   # Feature engineering
│   │   ├── model_agent.py     # Model training & evaluation
│   │   └── deployment_agent.py # Model deployment
│   ├── api/              # REST API
│   │   └── main.py
│   ├── utils/            # Utilities
│   │   ├── data_validation.py
│   │   ├── feature_engineering.py
│   │   ├── model_evaluation.py
│   │   └── logging_config.py
│   ├── config.py         # Configuration management
│   └── pipeline.py       # Main pipeline orchestrator
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Processed data
│   └── sample/           # Sample datasets
├── models/               # Trained models
├── deployments/          # Deployment artifacts
├── logs/                 # Pipeline logs
├── notebooks/            # Jupyter notebooks
├── tests/                # Test files
├── docker/               # Docker configurations
├── main.py               # CLI entry point
├── requirements.txt      # Dependencies
└── config.json          # Default configuration
```

## Pipeline Stages

### 1. Data Ingestion & Validation
- Loads data from various formats (CSV, Excel, JSON, Parquet)
- Validates data quality and structure
- Performs basic data profiling

### 2. Data Cleaning
- Handles missing values intelligently
- Removes duplicates and outliers
- Fixes data type inconsistencies

### 3. Feature Engineering
- Encodes categorical variables
- Creates datetime features
- Generates interaction and polynomial features
- Performs feature selection
- Scales numeric features

### 4. Model Training
- Trains multiple ML models:
  - Random Forest
  - Gradient Boosting (XGBoost, LightGBM)
  - Logistic/Linear Regression
  - Support Vector Machines
  - Neural Networks
- Performs hyperparameter tuning
- Uses cross-validation for robust evaluation

### 5. Model Evaluation
- Comprehensive performance metrics
- Model interpretation and explainability
- Cross-model comparison
- Performance requirement validation

### 6. Deployment
- Generates FastAPI application
- Creates Docker configuration
- Sets up monitoring and logging
- Provides deployment scripts

## Configuration

The pipeline can be configured through:

1. **Configuration File** (`config.json`)
2. **Environment Variables**
3. **Command Line Arguments**

### Key Configuration Options

```json
{
  "training": {
    "TEST_SIZE": 0.2,
    "CV_FOLDS": 5,
    "MAX_TRAINING_TIME": 3600
  },
  "thresholds": {
    "MIN_ACCURACY": 0.7,
    "MIN_F1_SCORE": 0.65
  },
  "feature_engineering": {
    "MAX_FEATURES": 50,
    "CORRELATION_THRESHOLD": 0.9
  }
}
```

## Model Deployment

The pipeline automatically generates:

- **FastAPI Application**: Production-ready API with automatic documentation
- **Docker Configuration**: Complete containerization setup
- **Monitoring Setup**: Prometheus metrics and Grafana dashboards
- **API Documentation**: Interactive Swagger/ReDoc documentation

### Deployment Structure

```
deployments/your_project/
├── app/
│   ├── main.py           # FastAPI application
│   ├── model.py          # Model inference module
│   ├── schemas.py        # Pydantic models
│   └── config.py         # Application configuration
├── artifacts/
│   ├── model.pkl         # Trained model
│   ├── model_metadata.json
│   └── label_encoder.pkl (if needed)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── docs/
    ├── README.md
    ├── API.md
    └── DEPLOYMENT.md
```

## Monitoring

Built-in monitoring includes:

- **API Metrics**: Request count, latency, error rates
- **Model Metrics**: Prediction confidence, drift detection
- **System Metrics**: CPU, memory, disk usage
- **Custom Dashboards**: Grafana visualizations
- **Alerting**: Automated alerts for issues

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the documentation
2. Review the logs in `logs/`
3. Run the test script: `python test_pipeline.py`
4. Open an issue on GitHub

## Examples

### Classification Example (Titanic)

```bash
python main.py --data-path data/sample/titanic.csv --target-column Survived
```

### Regression Example

```bash
python main.py --data-path data/sample/housing.csv --target-column price
```

### Custom Configuration

```bash
python main.py \
  --data-path data/my_data.csv \
  --target-column target \
  --config my_config.json \
  --project-name my_custom_project
```