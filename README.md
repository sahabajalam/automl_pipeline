# AutoML Pipeline

An automated machine learning pipeline project, currently in Phase One development. This phase focuses on building the core components for data analysis, feature engineering, model training, API serving, and MLOps integration.

## Project Overview

This project aims to create a complete AutoML system that can:
- Analyze and validate datasets automatically
- Perform intelligent feature engineering
- Train and optimize multiple ML models
- Serve models via a production API
- Track experiments and manage model lifecycle

## Current Status: Phase One

Phase One implements the foundational components as outlined in `phase_one_Technical Scope & Components.md`. Each component is developed incrementally over four weeks:

### Week 1: Data Analysis Agent (GenAI Core)
- Automated exploratory data analysis with LLM integration
- Schema detection, statistical insights, domain recognition, quality scoring

### Week 2: Feature Engineering Pipeline (Classical ML Foundation)
- Smart preprocessing and feature creation
- Context-aware missing value handling, dynamic scaling, feature generation

### Week 3: Multi-Algorithm Training Engine
- Intelligent model selection and training
- Hyperparameter optimization, performance evaluation, model comparison

### Week 4: Production API Gateway (Deployment Foundation)
- RESTful interface for model serving
- Async processing, multi-format support, job management

### Week 5: Basic MLOps Integration
- Experiment tracking and model lifecycle management
- MLflow integration, model registry, artifact management

## Project Structure

```
automl_pipeline/
├── config.json                    # Configuration settings
├── pyproject.toml                 # Project dependencies and metadata
├── phase_one_Technical Scope & Components.md  # Phase One specification
├── README.md                      # This file
└── phase_one_component/           # Phase One component modules
    ├── __init__.py
    ├── 1_data_analysis_agent.py   # Data analysis functions
    ├── 2_feature_engineering_pipeline.py  # Feature engineering functions
    ├── 3_multi_algorithm_training.py      # Training engine functions
    ├── 4_production_api_gateway.py        # API gateway functions
    └── 5_basic_mlops_integration.py       # MLOps functions
```

## Setup and Installation

### Prerequisites
- Python 3.10+
- pip or conda for package management

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/sahabajalam/automl_pipeline.git
   cd automl_pipeline
   ```

2. Install dependencies:
   ```bash
   pip install -e .
   ```
   Or if using conda:
   ```bash
   conda env create -f environment.yml  # if available
   ```

3. Verify installation:
   ```bash
   python -c "import phase_one_component; print('Installation successful')"
   ```

## Usage

Currently, the project contains function stubs for each Phase One component. Implementation will be added progressively.

### Example: Importing Components

```python
from phase_one_component import (
    data_analysis_agent,
    feature_engineering_pipeline,
    multi_algorithm_training,
    production_api_gateway,
    basic_mlops_integration
)

# Example usage (stubs only - implementations coming)
import pandas as pd
df = pd.read_csv('your_data.csv')

# Analyze data schema
schema = data_analysis_agent.analyze_schema(df)
print(schema)
```

## Development Roadmap

### Phase One Deliverables (Current)
- [ ] Week 1: Functional data analysis agent with JSON outputs
- [ ] Week 2: Complete feature engineering pipeline
- [ ] Week 3: Multi-algorithm training framework
- [ ] Week 4: Production API with async processing
- [ ] Week 5: MLOps integration with MLflow

### Future Phases
- Phase Two: Integration and testing
- Phase Three: Advanced features (LLM enhancements, distributed training)
- Phase Four: Production deployment and monitoring

## Configuration

The pipeline uses `config.json` for configuration. Current settings include basic parameters for each component.

## Contributing

1. This is an early-stage project - contributions welcome for Phase One implementation
2. Follow the component specifications in `phase_one_Technical Scope & Components.md`
3. Add tests for new functionality
4. Update this README as components are implemented

## License

MIT License - see LICENSE file for details.

## Contact

Repository: https://github.com/sahabajalam/automl_pipeline