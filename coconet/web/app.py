"""FastAPI application for Coconet harmonizer."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from ..harmonize import Harmonizer
from .routes import router

# Global harmonizer instance
harmonizer: Harmonizer | None = None


def get_harmonizer() -> Harmonizer:
    """Get the global harmonizer instance."""
    if harmonizer is None:
        raise RuntimeError("Harmonizer not initialized")
    return harmonizer


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize harmonizer on startup."""
    global harmonizer

    # Check for checkpoint
    checkpoint_path = os.environ.get(
        "COCONET_CHECKPOINT",
        "checkpoints/best_model.pt"
    )

    if Path(checkpoint_path).exists():
        print(f"Loading model from {checkpoint_path}")
        harmonizer = Harmonizer(checkpoint_path=checkpoint_path)
    else:
        print("Warning: No checkpoint found, using untrained model")
        harmonizer = Harmonizer()

    yield

    # Cleanup
    harmonizer = None


app = FastAPI(
    title="Coconet Harmonizer",
    description="Generate Bach-style 4-part harmonizations from melodies",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS for frontend dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Vite dev server
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api")

# Serve static files for production (built React app)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
