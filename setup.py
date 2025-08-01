#!/usr/bin/env python3
"""
Setup script for the Autonomous ML Pipeline
"""
import os
import sys
import subprocess
from pathlib import Path

def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/sample",
        "models",
        "checkpoints",
        "deployments",
        "logs",
        "notebooks"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {directory}")

def check_python_version():
    """Check Python version"""
    if sys.version_info < (3, 9):
        print("❌ Python 3.9 or higher is required")
        return False
    
    print(f"✅ Python version: {sys.version}")
    return True

def install_dependencies():
    """Install dependencies"""
    try:
        print("📦 Installing dependencies...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def setup_mlflow():
    """Setup MLflow tracking"""
    try:
        print("🔬 Setting up MLflow...")
        # Create MLflow directory
        Path("mlruns").mkdir(exist_ok=True)
        print("✅ MLflow setup complete")
        return True
    except Exception as e:
        print(f"❌ MLflow setup failed: {e}")
        return False

def validate_setup():
    """Validate the setup"""
    try:
        print("🧪 Validating setup...")
        
        # Try importing key modules
        import pandas
        import sklearn
        import fastapi
        
        print("✅ Core dependencies available")
        
        # Check if sample data exists
        sample_data = Path("data/sample/titanic.csv")
        if sample_data.exists():
            print("✅ Sample data available")
        else:
            print("⚠️  Sample data not found - some tests may fail")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def main():
    """Main setup function"""
    print("🚀 Setting up Autonomous ML Pipeline")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    create_directories()
    
    # Install dependencies
    if not install_dependencies():
        print("\n💡 Try installing manually:")
        print("pip install -r requirements.txt")
        sys.exit(1)
    
    # Setup MLflow
    setup_mlflow()
    
    # Validate setup
    if validate_setup():
        print("\n🎉 Setup completed successfully!")
        print("\nNext steps:")
        print("1. Run the test: python test_pipeline.py")
        print("2. Try the CLI: python main.py --data-path data/sample/titanic.csv --target-column Survived")
        print("3. Start the API: python src/api/main.py")
    else:
        print("\n❌ Setup validation failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
