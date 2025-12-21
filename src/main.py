"""FastAPI application entry point."""

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .api import router
from .config import settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    print(f"Research Tool starting on {settings.host}:{settings.port}")
    print(f"LLM API: {settings.llm_api_base}")
    print(f"SearXNG: {settings.searxng_host}")

    yield

    # Shutdown
    print("Research Tool shutting down")


app = FastAPI(
    title="Research Tool",
    description="Lightweight hybrid research tool with multi-source search and LLM synthesis",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api/v1", tags=["research"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API info."""
    return {
        "name": "Research Tool",
        "version": "1.0.0",
        "docs": "/docs",
        "endpoints": {
            # Core endpoints
            "health": "/api/v1/health",
            "search": "/api/v1/search",
            "research": "/api/v1/research",
            # Perplexity-equivalent endpoints
            "ask": "/api/v1/ask",           # mirrors perplexity_ask
            "discover": "/api/v1/discover", # mirrors perplexity_search (EXPLORATORY)
            "synthesize": "/api/v1/synthesize",  # mirrors perplexity_research (SYNTHESIS)
            "reason": "/api/v1/reason",     # mirrors perplexity_reason
        },
    }


def run():
    """Run the application with uvicorn."""
    import uvicorn
    uvicorn.run(
        "src.main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    run()
