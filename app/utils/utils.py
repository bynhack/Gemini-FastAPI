"""工具函数模块"""

import base64
import os
import re
import tempfile
from typing import List, Tuple

from ..models import Message


def correct_markdown(md_text: str) -> str:
    """
    修正Markdown文本，移除Google搜索链接包装器，并根据显示文本简化目标URL。

    Args:
        md_text: 原始Markdown文本

    Returns:
        修正后的Markdown文本
    """

    def simplify_link_target(text_content: str) -> str:
        """简化链接目标"""
        match_colon_num = re.match(r"([^:]+:\d+)", text_content)
        if match_colon_num:
            return match_colon_num.group(1)
        return text_content

    def replacer(match: re.Match) -> str:
        """链接替换器"""
        outer_open_paren = match.group(1)
        display_text = match.group(2)

        new_target_url = simplify_link_target(display_text)
        new_link_segment = f"[`{display_text}`]({new_target_url})"

        if outer_open_paren:
            return f"{outer_open_paren}{new_link_segment})"
        else:
            return new_link_segment

    # 修复Google搜索链接
    pattern = r"(\()?\[`([^`]+?)`\]\((https://www.google.com/search\?q=)(.*?)(?<!\\)\)\)*(\))?"
    fixed_google_links = re.sub(pattern, replacer, md_text)

    # 修复包装的markdown链接
    pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
    return re.sub(pattern, r"\1", fixed_google_links)


def prepare_conversation(messages: List[Message]) -> Tuple[str, List[str]]:
    """
    将OpenAI格式的消息转换为对话格式

    Args:
        messages: 消息列表

    Returns:
        (对话文本, 临时文件列表)
    """
    conversation = ""
    temp_files = []

    for msg in messages:
        if isinstance(msg.content, str):
            # 字符串内容处理
            if msg.role == "system":
                conversation += f"System: {msg.content}\n\n"
            elif msg.role == "user":
                conversation += f"Human: {msg.content}\n\n"
            elif msg.role == "assistant":
                conversation += f"Assistant: {msg.content}\n\n"
        else:
            # 混合内容处理
            if msg.role == "user":
                conversation += "Human: "
            elif msg.role == "system":
                conversation += "System: "
            elif msg.role == "assistant":
                conversation += "Assistant: "

            for item in msg.content:
                if item.type == "text":
                    conversation += item.text or ""
                elif item.type == "image_url" and item.image_url:
                    # 处理图片
                    image_url = item.image_url.get("url", "")
                    if image_url.startswith("data:image/"):
                        # 处理base64编码的图片
                        try:
                            # 提取base64部分
                            base64_data = image_url.split(",")[1]
                            image_data = base64.b64decode(base64_data)

                            # 创建临时文件保存图片
                            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                                tmp.write(image_data)
                                temp_files.append(tmp.name)
                        except Exception:
                            # 这里应该用日志记录，但为了避免循环导入，暂时忽略
                            pass

            conversation += "\n\n"

    # 添加最终的助手提示
    conversation += "Assistant: "

    return conversation, temp_files


def cleanup_temp_files(temp_files: List[str]) -> None:
    """
    清理临时文件

    Args:
        temp_files: 临时文件路径列表
    """
    for temp_file in temp_files:
        try:
            os.unlink(temp_file)
        except Exception:
            # 忽略删除失败的情况
            pass


def format_response_text(response) -> str:
    """
    格式化响应文本

    Args:
        response: Gemini API响应

    Returns:
        格式化后的文本
    """
    reply_text = ""

    if hasattr(response, "thoughts") and response.thoughts:
        reply_text += f"<think>{response.thoughts}</think>"

    if hasattr(response, "text") and response.text:
        reply_text += response.text
    else:
        reply_text += str(response)

    # 清理转义字符
    reply_text = reply_text.replace("&lt;", "<").replace("\\<", "<").replace("\\_", "_").replace("\\>", ">")

    # 修正markdown
    reply_text = correct_markdown(reply_text)

    return reply_text


def estimate_tokens(text: str) -> int:
    """
    简单的token估算（粗略估算）

    Args:
        text: 输入文本

    Returns:
        估算的token数量
    """
    # 简单的token估算：大约4个字符 = 1个token
    return len(text.split())
