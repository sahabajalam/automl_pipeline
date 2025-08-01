#!/usr/bin/env python3
"""
Test script for the Autonomous ML Pipeline
"""
import asyncio
import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

async def test_pipeline():
    """Test the complete pipeline"""
    try:
        from src.pipeline import AutonomousMLPipeline
        from src.utils.logging_config import setup_logging
        
        # Setup logging
        setup_logging(log_level="INFO")
        
        print("🚀 Testing Autonomous ML Pipeline")
        print("=" * 50)
        
        # Initialize pipeline
        print("1. Initializing pipeline...")
        pipeline = AutonomousMLPipeline()
        print("✅ Pipeline initialized")
        
        # Test data path
        data_path = "data/sample/titanic.csv"
        if not os.path.exists(data_path):
            print(f"❌ Test data not found at {data_path}")
            return False
        
        print(f"2. Using test data: {data_path}")
        
        # Run pipeline
        print("3. Running pipeline...")
        result = await pipeline.run_pipeline(
            data_path=data_path,
            target_column="Survived",
            project_name="test_titanic_pipeline"
        )
        
        # Check results
        if result.get('status') == 'failed':
            print(f"❌ Pipeline failed: {result.get('error')}")
            return False
        
        print("✅ Pipeline completed successfully!")
        print(f"   Project: {result.get('project_name')}")
        
        best_model = result.get('best_model', {})
        if best_model:
            print(f"   Best model: {best_model.get('model_name', 'Unknown')}")
            performance = best_model.get('performance', {})
            if performance:
                print(f"   Performance: {performance}")
        
        deployment_info = result.get('deployment_info', {})
        if deployment_info:
            print(f"   Deployment: {deployment_info.get('status', 'Not deployed')}")
        
        print("\n🎉 All tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main test function"""
    success = asyncio.run(test_pipeline())
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
