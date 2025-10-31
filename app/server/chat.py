import uuid
from datetime import datetime, timezone
from pathlib import Path

import orjson
import base64
from fastapi import APIRouter, Depends, HTTPException, Request, status
import httpx
from fastapi.responses import StreamingResponse
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from loguru import logger

from ..models import (
    ChatCompletionRequest,
    ConversationInStore,
    ImagesGenerateRequest,
    ImagesGenerateResponse,
    ImageData,
    Message,
    ModelData,
    ModelListResponse,
)
from ..services import (
    GeminiClientPool,
    GeminiClientWrapper,
    LMDBConversationStore,
)
from ..utils import g_config
from ..utils.helper import estimate_tokens
from .middleware import get_temp_dir, verify_api_key

# Maximum characters Gemini Web can accept in a single request (configurable)
MAX_CHARS_PER_REQUEST = int(g_config.gemini.max_chars_per_request * 0.9)

CONTINUATION_HINT = "\n(More messages to come, please reply with just 'ok.')"


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
    pool = GeminiClientPool()
    db = LMDBConversationStore()
    model = Model.from_name(request.model)

    if len(request.messages) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one message is required in the conversation.",
        )

    # Check if conversation is reusable
    session, client, remaining_messages = _find_reusable_session(db, pool, model, request.messages)

    if session:
        # Prepare the model input depending on how many turns are missing.
        if len(remaining_messages) == 1:
            model_input, files = await GeminiClientWrapper.process_message(
                remaining_messages[0], tmp_dir, tagged=False
            )
        else:
            model_input, files = await GeminiClientWrapper.process_conversation(
                remaining_messages, tmp_dir
            )
        logger.debug(
            f"Reused session {session.metadata} - sending {len(remaining_messages)} new messages."
        )
    else:
        # Start a new session and concat messages into a single string
        try:
            client = pool.acquire()
            session = client.start_chat(model=model)
            model_input, files = await GeminiClientWrapper.process_conversation(
                request.messages, tmp_dir
            )
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
        except Exception as e:
            logger.exception(f"Error in preparing conversation: {e}")
            raise
        logger.debug("New session started.")

    # Generate response
    try:
        assert session and client, "Session and client not available"
        logger.debug(
            f"Client ID: {client.id}, Input length: {len(model_input)}, files count: {len(files)}"
        )
        response = await _send_with_split(session, model_input, files=files)
    except Exception as e:
        logger.exception(f"Error generating content from Gemini API: {e}")
        raise

    # Format the response from API
    model_output = GeminiClientWrapper.extract_output(response, include_thoughts=True)
    stored_output = GeminiClientWrapper.extract_output(response, include_thoughts=False)
    image_urls = GeminiClientWrapper.extract_images(response)

    # After formatting, persist the conversation to LMDB
    try:
        last_message = Message(role="assistant", content=stored_output)
        cleaned_history = db.sanitize_assistant_messages(request.messages)
        conv = ConversationInStore(
            model=model.model_name,
            client_id=client.id,
            metadata=session.metadata,
            messages=[*cleaned_history, last_message],
        )
        key = db.store(conv)
        logger.debug(f"Conversation saved to LMDB with key: {key}")
    except Exception as e:
        # We can still return the response even if saving fails
        logger.warning(f"Failed to save conversation to LMDB: {e}")

    # Return with streaming or standard response
    completion_id = f"chatcmpl-{uuid.uuid4()}"
    timestamp = int(datetime.now(tz=timezone.utc).timestamp())
    if request.stream:
        return _create_streaming_response(
            model_output,
            completion_id,
            timestamp,
            request.model,
            request.messages,
        )
    else:
        return _create_standard_response(
            model_output, completion_id, timestamp, request.model, request.messages, image_urls
        )


@router.post("/v1/images/generations", response_model=ImagesGenerateResponse)
async def create_image_generation(
    request: ImagesGenerateRequest,
    http_request: Request,
    api_key: str = Depends(verify_api_key),
):
    pool = GeminiClientPool()
    model = Model.from_name(request.model)

    try:
        client = pool.acquire()
        session = client.start_chat(model=model)
        response = await client.generate_content(request.prompt, chat=session, model=model)
    except Exception as e:
        logger.exception(f"Error generating image from Gemini API: {e}")
        raise HTTPException(status_code=500, detail="Image generation failed")

    # 获取图片对象列表
    image_objects = GeminiClientWrapper.get_image_objects(response)
    if request.n and request.n > 0:
        image_objects = image_objects[: request.n]

    created = int(datetime.now(tz=timezone.utc).timestamp())

    # 保存图片到本地并返回 URL
    images_dir = Path("data/images")
    images_dir.mkdir(parents=True, exist_ok=True)

    data: list[ImageData] = []
    for i, image in enumerate(image_objects):
        try:
            # 生成唯一文件名
            timestamp = int(datetime.now(tz=timezone.utc).timestamp())
            filename = f"image_{timestamp}_{i}.png"
            filepath = images_dir / filename

            # 保存图片
            await image.save(path=str(images_dir), filename=filename, verbose=True)

            # 构建本地服务 URL
            if http_request:
                # 从请求对象获取基础 URL
                scheme = http_request.url.scheme
                host = http_request.url.hostname
                port = http_request.url.port
                if port:
                    base_url = f"{scheme}://{host}:{port}"
                else:
                    base_url = f"{scheme}://{host}"
            else:
                # 回退到使用配置
                scheme = "https" if g_config.server.https.enabled else "http"
                host = g_config.server.host
                port = g_config.server.port
                if host == "0.0.0.0":
                    host = "localhost"
                base_url = f"{scheme}://{host}:{port}"
            
            local_url = f"{base_url}/static/images/{filename}"

            if request.response_format == "b64_json":
                # 读取保存的文件并转换为 base64
                if filepath.exists():
                    with open(filepath, "rb") as f:
                        img_data = f.read()
                        b64 = base64.b64encode(img_data).decode("utf-8")
                        data.append(ImageData(b64_json=b64))
            else:
                data.append(ImageData(url=local_url))

        except Exception as e:
            logger.exception(f"Error saving image {i}: {e}")
            continue

    return ImagesGenerateResponse(created=created, data=data)


def _text_from_message(message: Message) -> str:
    """Return text content from a message for token estimation."""
    if isinstance(message.content, str):
        return message.content
    return "\n".join(
        item.text or "" for item in message.content if getattr(item, "type", "") == "text"
    )


def _find_reusable_session(
    db: LMDBConversationStore,
    pool: GeminiClientPool,
    model: Model,
    messages: list[Message],
) -> tuple[ChatSession | None, GeminiClientWrapper | None, list[Message]]:
    """Find an existing chat session that matches the *longest* prefix of
    ``messages`` **whose last element is an assistant/system reply**.

    Rationale
    ---------
    When a reply was generated by *another* server instance, the local LMDB may
    only contain an older part of the conversation.  However, as long as we can
    line-up **any** earlier assistant/system response, we can restore the
    corresponding Gemini session and replay the *remaining* turns locally
    (including that missing assistant reply and the subsequent user prompts).

    The algorithm therefore walks backwards through the history **one message at
    a time**, each time requiring the current tail to be assistant/system before
    querying LMDB.  As soon as a match is found we recreate the session and
    return the untouched suffix as ``remaining_messages``.
    """

    if len(messages) < 2:
        return None, None, messages

    # Start with the full history and iteratively trim from the end.
    search_end = len(messages)
    while search_end >= 2:
        search_history = messages[:search_end]

        # Only try to match if the last stored message would be assistant/system.
        if search_history[-1].role in {"assistant", "system"}:
            try:
                if conv := db.find(model.model_name, search_history):
                    client = pool.acquire(conv.client_id)
                    session = client.start_chat(metadata=conv.metadata, model=model)
                    remain = messages[search_end:]
                    return session, client, remain
            except Exception as e:
                logger.warning(f"Error checking LMDB for reusable session: {e}")
                break

        # Trim one message and try again.
        search_end -= 1

    return None, None, messages


async def _send_with_split(session: ChatSession, text: str, files: list[Path | str] | None = None):
    """Send text to Gemini, automatically splitting into multiple batches if it is
    longer than ``MAX_CHARS_PER_REQUEST``.

    Every intermediate batch (that is **not** the last one) is suffixed with a hint
    telling Gemini that more content will come, and it should simply reply with
    "ok". The final batch carries any file uploads and the real user prompt so
    that Gemini can produce the actual answer.
    """
    if len(text) <= MAX_CHARS_PER_REQUEST:
        # No need to split - a single request is fine.
        return await session.send_message(text, files=files)
    hint_len = len(CONTINUATION_HINT)
    chunk_size = MAX_CHARS_PER_REQUEST - hint_len

    chunks: list[str] = []
    pos = 0
    total = len(text)
    while pos < total:
        end = min(pos + chunk_size, total)
        chunk = text[pos:end]
        pos = end

        # If this is NOT the last chunk, add the continuation hint.
        if end < total:
            chunk += CONTINUATION_HINT
        chunks.append(chunk)

    # Fire off all but the last chunk, discarding the interim "ok" replies.
    for chk in chunks[:-1]:
        try:
            await session.send_message(chk)
        except Exception as e:
            logger.exception(f"Error sending chunk to Gemini: {e}")
            raise

    # The last chunk carries the files (if any) and we return its response.
    return await session.send_message(chunks[-1], files=files)


def _create_streaming_response(
    model_output: str,
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
) -> StreamingResponse:
    """Create streaming response with `usage` calculation included in the final chunk."""

    # Calculate token usage
    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    completion_tokens = estimate_tokens(model_output)
    total_tokens = prompt_tokens + completion_tokens

    async def generate_stream():
        # Send start event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}],
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Stream output text in chunks for efficiency
        chunk_size = 32
        for i in range(0, len(model_output), chunk_size):
            chunk = model_output[i : i + chunk_size]
            data = {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created_time,
                "model": model,
                "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}],
            }
            yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"

        # Send end event
        data = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created_time,
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        yield f"data: {orjson.dumps(data).decode('utf-8')}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate_stream(), media_type="text/event-stream")


def _create_standard_response(
    model_output: str,
    completion_id: str,
    created_time: int,
    model: str,
    messages: list[Message],
    image_urls: list[str] | None = None,
) -> dict:
    """Create standard response"""
    # Calculate token usage
    prompt_tokens = sum(estimate_tokens(_text_from_message(msg)) for msg in messages)
    completion_tokens = estimate_tokens(model_output)
    total_tokens = prompt_tokens + completion_tokens

    # 组装消息内容：如果包含图片，则返回多模态数组（文本 + image_url 项）
    if image_urls:
        content: list[dict] = []
        if model_output:
            content.append({"type": "text", "text": model_output})
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})
        message_content = content
    else:
        message_content = model_output

    result = {
        "id": completion_id,
        "object": "chat.completion",
        "created": created_time,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": message_content},
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
