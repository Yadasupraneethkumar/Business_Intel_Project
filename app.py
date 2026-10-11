from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from dotenv import load_dotenv

#Database initialization scripts
from database.metrics import get_metrics
from database.create_table import create_tables

# Official Application Routes
from routes.ingestion import router as ingestion_router
from routes.chat import router as chat_router
from routes.dashboard import router as dashboard_router
from routes.health import router as health_router
from contextlib import asynccontextmanager
load_dotenv()


@asynccontextmanager
async def lifespan(app: FastAPI):
    create_tables()
    yield


app = FastAPI(
    title="AI-Powered Investor Intelligence Platform",
    lifespan=lifespan
)

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)

templates = Jinja2Templates(
    directory="templates"
)

app.include_router(
    ingestion_router,
    prefix="/api",
    tags=["Ingestion"]
)

app.include_router(
    chat_router,
    prefix="/api",
    tags=["Chat"]
)

app.include_router(
    dashboard_router,
    prefix="/api",
    tags=["Dashboard"]
)

app.include_router(
    health_router,
    prefix="/api",
    tags=["Health"]
)

@app.get("/api/dashboard")
def dashboard(request: Request):
    """
    Render dashboard UI.
    """
    metrics = get_metrics()

    return templates.TemplateResponse(
        request=request,
        name="dashboard.html",
        context={
            "metrics": metrics,
            "total_companies": len(metrics),
            "total_reports": len(metrics)
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000
    )