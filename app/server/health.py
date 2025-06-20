from fastapi import APIRouter
from loguru import logger

from ..services.client import SingletonGeminiClient

router = APIRouter()


@router.get("/health")
async def health_check():
    client = SingletonGeminiClient()

    if not client.running:
        try:
            await client.init()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
            }

    return {"status": "healthy"}
