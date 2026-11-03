"""
Hospital FAQ Chatbot - Application Entry Point
Assembles the modular FastAPI backend, registers routers, and mounts the web frontend.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.routers import chat, hospital, system

# Initialize FastAPI application
app = FastAPI(
    title=settings.APP_TITLE,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for browser access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register Routers
app.include_router(chat.router)
app.include_router(hospital.router)
app.include_router(system.router)

# Mount Static Files
if settings.STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(settings.STATIC_DIR)), name="static")


@app.get("/", tags=["Frontend"])
async def serve_index():
    """Serves the web chat user interface."""
    index_file = settings.STATIC_DIR / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return JSONResponse({
        "message": "Hospital FAQ Chatbot API is running.",
        "api_documentation": "/docs"
    })


if __name__ == "__main__":
    import uvicorn
    print(f"\n=======================================================")
    print(f" {settings.APP_TITLE} v{settings.APP_VERSION}")
    print(f" Web UI:      http://{settings.HOST}:{settings.PORT}")
    print(f" API Docs:    http://{settings.HOST}:{settings.PORT}/docs")
    print(f" Model:       OpenAI ({settings.OPENAI_MODEL})")
    print(f"=======================================================\n")
    uvicorn.run("main:app", host=settings.HOST, port=settings.PORT, reload=True)
