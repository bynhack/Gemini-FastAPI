"""Pydantic 数据模型定义"""

from typing import Dict, List, Optional, Union

from pydantic import BaseModel


class ContentItem(BaseModel):
    """内容项模型"""

    type: str
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None


class Message(BaseModel):
    """消息模型"""

    role: str
    content: Union[str, List[ContentItem]]
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """聊天完成请求模型"""

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
    """选择模型"""

    index: int
    message: Message
    finish_reason: str


class Usage(BaseModel):
    """使用统计模型"""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """聊天完成响应模型"""

    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class ModelData(BaseModel):
    """模型数据模型"""

    id: str
    object: str = "model"
    created: int
    owned_by: str = "google"


class ModelListResponse(BaseModel):
    """模型列表模型"""

    object: str = "list"
    data: List[ModelData]


class ErrorResponse(BaseModel):
    """错误响应模型"""

    error: Dict[str, str]
