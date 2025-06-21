import json
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from gemini_webapi.constants import Model
from loguru import logger

from ..models import ChatCompletionRequest, ModelData, ModelListResponse
from ..services.client import SingletonGeminiClient
from ..utils.utils import (
    cleanup_temp_files,
    estimate_tokens,
    format_response_text,
    prepare_conversation,
)
from .middleware import verify_api_key

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
    request: ChatCompletionRequest, api_key: str = Depends(verify_api_key)
):
    client = SingletonGeminiClient()
    model = Model.from_name(request.model)

    # 转换消息为对话格式
    conversation, temp_files = prepare_conversation(request.messages)
    logger.debug(f"Prepared conversation length: {len(conversation)}")
    logger.debug(f"Number of temp files: {len(temp_files)}")

    try:
        # 生成响应
        logger.info("Sending request to Gemini...")
        if temp_files:
            # 包含文件的请求
            response = await client.generate_content(
                conversation, files=[Path(f) for f in temp_files], model=model
            )
        else:
            # 纯文本请求
            response = await client.generate_content(conversation, model=model)

        # 格式化响应文本
        reply_text = format_response_text(response)

        if not reply_text or reply_text.strip() == "":
            logger.warning("Empty response received from Gemini")
            reply_text = "服务器返回了空响应。请检查 Gemini API 凭据是否有效。"

        # 生成响应ID和时间戳
        completion_id = f"chatcmpl-{uuid.uuid4()}"
        created_time = int(time.time())

        # 检查是否需要流式响应
        if request.stream:
            return _create_streaming_response(
                reply_text, completion_id, created_time, request.model
            )
        else:
            return _create_standard_response(
                reply_text, completion_id, created_time, request.model, conversation
            )

    finally:
        # 清理临时文件
        cleanup_temp_files(temp_files)


def _create_streaming_response(
    reply_text: str, completion_id: str, created_time: int, model: str
) -> StreamingResponse:
    """创建流式响应"""

    async def generate_stream():
        # 发送开始事件
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {json.dumps(data)}\n\n"

        # 流式输出文本
        for char in reply_text:
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": char}, "finish_reason": None}],
            }
            yield f"data: {json.dumps(data)}\n\n"

        # 发送结束事件
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
    reply_text: str, completion_id: str, created_time: int, model: str, conversation: str
) -> dict:
    """创建标准响应"""
    # 计算token使用量
    prompt_tokens = estimate_tokens(conversation)
    completion_tokens = estimate_tokens(reply_text)
    total_tokens = prompt_tokens + completion_tokens

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": reply_text},
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
