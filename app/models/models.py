from typing import Dict, List, Literal, Optional, Union

from pydantic import BaseModel


class ContentItem(BaseModel):
    """Content item model"""

    type: Literal["text", "image_url", "file", "input_audio"]
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    file: Optional[Dict[str, str]] = None


class Message(BaseModel):
    """Message model"""

    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """Chat completion request model"""

    model: str
    messages: List[Message]
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stream: Optional[bool] = False
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = 0
    frequency_penalty: Optional[float] = 0
    user: Optional[str] = None


class Choice(BaseModel):
    """Choice model"""

    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """Usage statistics model"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """Chat completion response model"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelData(BaseModel):
    """Model data model"""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"


class ModelListResponse(BaseModel):
    """Model list model"""

    object: str = "list"
    data: List[ModelData]


class ErrorResponse(BaseModel):
    """Error response model"""

    error: Dict[str, str]
