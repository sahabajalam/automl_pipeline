# src/pipeline.py
from langgraph.graph import StateGraph
from langgraph.checkpoint.memory import MemorySaver
from typing import TypedDict, Annotated, Sequence, Optional, List
import pandas as pd
from datetime import datetime
import logging
import os
from src.config import get_config
from src.utils.logging_config import setup_logging, get_logger

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PipelineState(TypedDict, total=False):
    """State shared across all agents"""
    # Input
    data_path: str
    target_column: str
    project_name: str
    
    # Data Processing
    raw_data: Optional[pd.DataFrame]
    processed_data: Optional[pd.DataFrame]
    cleaned_data: Optional[pd.DataFrame]
    validation_report: Optional[dict]
    feature_report: Optional[dict]
    data_info: Optional[dict]
    cleaning_report: Optional[dict]
    
    # Model Training
    trained_models: Optional[dict]
    best_model: Optional[dict]
    evaluation_results: Optional[dict]
    training_report: Optional[dict]
    
    # Deployment
    deployment_info: Optional[dict]
    deployment_url: Optional[str]
    model_id: Optional[str]
    monitoring_info: Optional[dict]
    
    # Workflow
    current_step: str
    next_action: str
    errors: List[str]
    execution_log: List[str]

class AutonomousMLPipeline:
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        """Initialize the autonomous ML pipeline"""
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Initialize checkpointer for state persistence
        self.checkpointer = MemorySaver()
        
        # Build the graph
        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile(checkpointer=self.checkpointer)
        
        logger.info("Autonomous ML Pipeline initialized successfully")
    
    def _build_graph(self) -> StateGraph:
        """Build the LangGraph workflow"""
        from src.agents.data_agent import DataIngestionAgent
        from src.agents.feature_agent import FeatureEngineeringAgent
        from src.agents.model_agent import ModelTrainingAgent
        from src.agents.deployment_agent import DeploymentAgent
        
        # Initialize agents
        data_agent = DataIngestionAgent()
        feature_agent = FeatureEngineeringAgent()
        model_agent = ModelTrainingAgent()
        deployment_agent = DeploymentAgent()
        
        # Create the graph
        workflow = StateGraph(PipelineState)
        
        # Add nodes
        workflow.add_node("data_ingestion", data_agent.process)
        workflow.add_node("data_validation", data_agent.validate)
        workflow.add_node("data_cleaning", feature_agent.clean_data)
        workflow.add_node("feature_engineering", feature_agent.engineer_features)
        workflow.add_node("model_training", model_agent.train_models)
        workflow.add_node("model_evaluation", model_agent.evaluate_models)
        workflow.add_node("deployment", deployment_agent.deploy)
        workflow.add_node("monitoring_setup", deployment_agent.setup_monitoring)
        
        # Define the workflow edges
        workflow.set_entry_point("data_ingestion")
        
        # Sequential flow with conditional branching
        workflow.add_edge("data_ingestion", "data_validation")
        
        # Conditional: if validation fails, retry or error
        workflow.add_conditional_edges(
            "data_validation",
            self._route_after_validation,
            {
                "proceed": "data_cleaning",
                "retry": "data_ingestion",
                "error": "__end__"
            }
        )
        
        workflow.add_edge("data_cleaning", "feature_engineering")
        workflow.add_edge("feature_engineering", "model_training")
        workflow.add_edge("model_training", "model_evaluation")
        
        # Conditional: if model performance is poor, retrain or proceed
        workflow.add_conditional_edges(
            "model_evaluation",
            self._route_after_evaluation,
            {
                "deploy": "deployment",
                "retrain": "model_training",
                "re_engineer": "feature_engineering"
            }
        )
        
        workflow.add_edge("deployment", "monitoring_setup")
        workflow.add_edge("monitoring_setup", "__end__")
        
        return workflow
    
    def _route_after_validation(self, state: PipelineState) -> str:
        """Route based on data validation results"""
        validation_report = state.get("validation_report") or {}
        
        if validation_report.get("is_valid", False):
            return "proceed"
        elif validation_report.get("can_fix", False):
            return "retry"
        else:
            return "error"
    
    def _route_after_evaluation(self, state: PipelineState) -> str:
        """Route based on model evaluation results"""
        best_model = state.get("best_model") or {}
        performance = best_model.get("performance", {})
        
        # Define thresholds
        min_accuracy = 0.7
        min_f1 = 0.65
        
        accuracy = performance.get("accuracy", 0)
        f1_score = performance.get("f1_score", 0)
        
        if accuracy >= min_accuracy and f1_score >= min_f1:
            return "deploy"
        elif accuracy < 0.5:  # Very poor performance
            return "re_engineer"
        else:
            return "retrain"
    
    async def run_pipeline(self, 
                          data_path: str, 
                          target_column: str, 
                          project_name: Optional[str] = None) -> dict:
        """Execute the complete ML pipeline"""
        
        if project_name is None:
            project_name = f"ml_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Initial state
        initial_state = PipelineState(
            data_path=data_path,
            target_column=target_column,
            project_name=project_name,
            current_step="initialization",
            next_action="data_ingestion",
            errors=[],
            execution_log=[f"Pipeline started at {datetime.now()}"]
        )
        
        logger.info(f"Starting pipeline for project: {project_name}")
        
        try:
            # Run the compiled graph
            config = {"configurable": {"thread_id": project_name}}
            
            final_state = await self.compiled_graph.ainvoke(
                initial_state, 
                config=config
            )
            
            # Log completion
            final_state["execution_log"].append(
                f"Pipeline completed at {datetime.now()}"
            )
            
            logger.info(f"Pipeline completed successfully for {project_name}")
            return final_state
            
        except Exception as e:
            logger.error(f"Pipeline failed for {project_name}: {str(e)}")
            return {
                "status": "failed",
                "error": str(e),
                "project_name": project_name
            }
    
    def get_pipeline_status(self, project_name: str) -> dict:
        """Get current status of a running pipeline"""
        config = {"configurable": {"thread_id": project_name}}
        
        try:
            # Get current state from checkpointer
            state = self.checkpointer.get(config)
            if state is None:
                return {"status": "not_found"}
            
            return {
                "status": "running" if state.get("next_action") != "completed" else "completed",
                "current_step": state.get("current_step"),
                "next_action": state.get("next_action"),
                "execution_log": state.get("execution_log", [])
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def list_projects(self) -> list:
        """List all pipeline projects"""
        # Implementation depends on checkpointer capabilities
        # For now, return empty list
        return []

# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def main():
        pipeline = AutonomousMLPipeline()
        
        result = await pipeline.run_pipeline(
            data_path="data/sample/titanic.csv",
            target_column="survived",
            project_name="titanic_survival_prediction"
        )
        
        print("Pipeline Result:", result)
    
    asyncio.run(main())