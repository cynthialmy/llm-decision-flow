"""FastAPI application main file."""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from src.api.routes import content, review, metrics

app = FastAPI(
    title="LLM Decision Flow API",
    description="Policy-Aware Misinformation Decision System",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(content.router, prefix="/api", tags=["content"])
app.include_router(review.router, prefix="/api", tags=["review"])
app.include_router(metrics.router, prefix="/api", tags=["metrics"])


@app.get("/")
def root():
    """Root endpoint."""
    return {"message": "LLM Decision Flow API", "version": "1.0.0"}


@app.get("/health")
def health():
    """Health check endpoint."""
    return {"status": "healthy"}
