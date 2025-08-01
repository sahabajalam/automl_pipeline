import asyncio
import argparse
import sys
from pathlib import Path
from src.pipeline import AutonomousMLPipeline
from src.utils.logging_config import setup_logging
from src.config import get_config

def main():
    """Main entry point for the AutoML Pipeline"""
    parser = argparse.ArgumentParser(description="Autonomous ML Pipeline")
    parser.add_argument("--data-path", required=True, help="Path to the dataset")
    parser.add_argument("--target-column", required=True, help="Name of the target column")
    parser.add_argument("--project-name", help="Name of the ML project")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--log-level", default="INFO", help="Logging level")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Load configuration
    config = get_config(args.config)
    
    # Validate data path exists
    if not Path(args.data_path).exists():
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
    
    async def run_pipeline():
        """Run the autonomous ML pipeline"""
        try:
            # Initialize pipeline
            pipeline = AutonomousMLPipeline()
            
            # Run the pipeline
            result = await pipeline.run_pipeline(
                data_path=args.data_path,
                target_column=args.target_column,
                project_name=args.project_name
            )
            
            # Print results
            if result.get('status') == 'failed':
                print(f"❌ Pipeline failed: {result.get('error')}")
                sys.exit(1)
            else:
                print("🎉 Pipeline completed successfully!")
                print(f"Project: {result.get('project_name')}")
                print(f"Best model: {result.get('best_model', {}).get('model_name', 'Unknown')}")
                print(f"Deployment URL: {result.get('deployment_info', {}).get('deployment_url', 'Not deployed')}")
                
        except Exception as e:
            print(f"❌ Pipeline failed with error: {str(e)}")
            sys.exit(1)
    
    # Run the async pipeline
    asyncio.run(run_pipeline())

if __name__ == "__main__":
    main()
