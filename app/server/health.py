from fastapi import APIRouter
from loguru import logger

from ..models import HealthCheckResponse
from ..services import LMDBConversationStore, SingletonGeminiClient

router = APIRouter()


@router.get("/health", response_model=HealthCheckResponse)
async def health_check():
    client = SingletonGeminiClient()
    db = LMDBConversationStore()

    if not client.running:
        try:
            await client.init()
        except Exception as e:
            logger.error(f"Failed to initialize Gemini client: {e}")
            return HealthCheckResponse(ok=False, error=str(e))

    stat = db.stats()
    if not stat:
        logger.error("Failed to retrieve LMDB conversation store stats")
        return HealthCheckResponse(ok=False, error="LMDB conversation store unavailable")

    return HealthCheckResponse(ok=True, storage=stat)
