from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from loguru import logger

from .server.chat import router as chat_router
from .server.health import router as health_router
from .server.middleware import add_cors_middleware, add_exception_handler
from .services.pool import GeminiClientPool
from .utils import g_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        pool = GeminiClientPool()
        await pool.init()
    except Exception as e:
        logger.exception(f"Failed to initialize Gemini clients: {e}")
        raise

    logger.success(f"Gemini clients initialized: {[c.id for c in pool.clients]}.")
    logger.success("Gemini API Server ready to serve requests.")
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="Gemini API Server",
        description="OpenAI-compatible API for Gemini Web",
        version="1.0.0",
        lifespan=lifespan,
    )

    add_cors_middleware(app)
    add_exception_handler(app)

    # 挂载静态文件服务（用于生成的图片）
    images_dir = Path("data/images")
    images_dir.mkdir(parents=True, exist_ok=True)
    app.mount("/static/images", StaticFiles(directory=str(images_dir)), name="images")

    app.include_router(health_router, tags=["Health"])
    app.include_router(chat_router, tags=["Chat"])

    return app
