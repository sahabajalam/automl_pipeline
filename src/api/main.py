from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, Optional, List
import logging
import uvicorn
from datetime import datetime
import sys
import os

# Add the parent directory to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import AutonomousMLPipeline
from src.utils.logging_config import setup_logging

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Autonomous ML Pipeline API",
    description="REST API for the Autonomous ML Pipeline",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance
pipeline = None

class PipelineRequest(BaseModel):
    data_path: str
    target_column: str
    project_name: Optional[str] = None

class PipelineResponse(BaseModel):
    status: str
    project_name: str
    message: str
    results: Optional[Dict[str, Any]] = None

class StatusResponse(BaseModel):
    status: str
    current_step: Optional[str] = None
    next_action: Optional[str] = None
    execution_log: Optional[List[str]] = None

@app.on_event("startup")
async def startup_event():
    """Initialize the pipeline on startup"""
    global pipeline
    try:
        pipeline = AutonomousMLPipeline()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {str(e)}")
        raise

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/pipeline/run", response_model=PipelineResponse)
async def run_pipeline(request: PipelineRequest, background_tasks: BackgroundTasks):
    """Start a new ML pipeline run"""
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        # Validate that data file exists
        import os
        if not os.path.exists(request.data_path):
            raise HTTPException(status_code=400, detail=f"Data file not found: {request.data_path}")
        
        # Run pipeline in background
        background_tasks.add_task(
            run_pipeline_background,
            request.data_path,
            request.target_column,
            request.project_name
        )
        
        project_name = request.project_name or f"ml_project_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        return PipelineResponse(
            status="started",
            project_name=project_name,
            message="Pipeline started successfully. Use /pipeline/status/{project_name} to check progress."
        )
        
    except Exception as e:
        logger.error(f"Failed to start pipeline: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

async def run_pipeline_background(data_path: str, target_column: str, project_name: Optional[str]):
    """Run pipeline in background"""
    global pipeline
    try:
        if pipeline is None:
            logger.error("Pipeline not initialized")
            return
            
        result = await pipeline.run_pipeline(
            data_path=data_path,
            target_column=target_column,
            project_name=project_name
        )
        logger.info(f"Pipeline completed for project: {result.get('project_name')}")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")

@app.get("/pipeline/status/{project_name}", response_model=StatusResponse)
async def get_pipeline_status(project_name: str):
    """Get the status of a running pipeline"""
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        status = pipeline.get_pipeline_status(project_name)
        
        if status.get("status") == "not_found":
            raise HTTPException(status_code=404, detail="Project not found")
        
        return StatusResponse(
            status=status.get("status", "unknown"),
            current_step=status.get("current_step"),
            next_action=status.get("next_action"),
            execution_log=status.get("execution_log", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get pipeline status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/pipeline/projects")
async def list_projects():
    """List all pipeline projects"""
    
    if pipeline is None:
        raise HTTPException(status_code=500, detail="Pipeline not initialized")
    
    try:
        projects = pipeline.list_projects()
        return {"projects": projects}
    except Exception as e:
        logger.error(f"Failed to list projects: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Autonomous ML Pipeline API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )