import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from gemini_webapi.constants import Model
from loguru import logger

from ..models import ChatCompletionRequest, ModelData, ModelListResponse
from ..services.client import SingletonGeminiClient
from ..utils.utils import (
    estimate_tokens,
)
from .middleware import get_temp_dir, verify_api_key

router = APIRouter()


@router.get("/v1/models", response_model=ModelListResponse)
async def list_models(api_key: str = Depends(verify_api_key)):
    now = int(datetime.now(tz=timezone.utc).timestamp())

    models = []
    for model in Model:
        m_name = model.model_name
        if not m_name or m_name == "unspecified":
            continue

        models.append(
            ModelData(
                id=m_name,
                created=now,
                owned_by="gemini-web",
            )
        )

    return ModelListResponse(data=models)


@router.post("/v1/chat/completions")
async def create_chat_completion(
    request: ChatCompletionRequest,
    api_key: str = Depends(verify_api_key),
    tmp_dir: Path = Depends(get_temp_dir),
):
    client = SingletonGeminiClient()
    model = Model.from_name(request.model)

    # Preprocess the messages
    try:
        conversation, files = await client.prepare(request.messages, tmp_dir)
        conversation = "\n".join(conversation)
        logger.debug(f"Conversation length: {len(conversation)}, files count: {len(files)}")
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except Exception as e:
        logger.exception(f"Error in preparing conversation: {e}")
        raise

    # Generate response
    try:
        response = await client.generate_content(conversation, files=files, model=model)
    except Exception as e:
        logger.exception(f"Error generating content from Gemini API: {e}")
        raise

    # Post process
    response_text = client.format_response(response)
    if not response_text or response_text.strip() == "":
        logger.warning("Empty response received from Gemini")
        response_text = "No response generated."

    completion_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(time.time())

    # Return with streaming or standard response
    if request.stream:
        return _create_streaming_response(response_text, completion_id, timestamp, request.model)
    else:
        return _create_standard_response(
            response_text, completion_id, timestamp, request.model, conversation
        )


def _create_streaming_response(
    response_text: str, completion_id: str, created_time: int, model: str
) -> StreamingResponse:
    """Create streaming response"""

    async def generate_stream():
        # Send start event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n"

        # Stream output text
        for char in response_text:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"

        # Send end event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        }
        yield f"data: {json.dumps(data)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_standard_response(
    response_text: str, completion_id: str, created_time: int, model: str, conversation: str
) -> dict:
    """Create standard response"""
    # Calculate token usage
    prompt_tokens = estimate_tokens(conversation)
    completion_tokens = estimate_tokens(response_text)
    total_tokens = prompt_tokens + completion_tokens

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": response_text},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }

    logger.debug(f"Response created with {total_tokens} total tokens")
    return result
