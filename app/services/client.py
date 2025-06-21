import re
from pathlib import Path

from gemini_webapi import GeminiClient, ModelOutput

from ..models import Message
from ..utils import g_config
from ..utils.singleton import Singleton
from ..utils.utils import add_tag, save_file_to_tempfile, save_url_to_tempfile


class SingletonGeminiClient(GeminiClient, metaclass=Singleton):
    def __init__(self, **kwargs):
        # TODO: Add proxy support if needed
        super().__init__(
            secure_1psid=g_config.gemini.secure_1psid,
            secure_1psidts=g_config.gemini.secure_1psidts,
            **kwargs,
        )

    async def init(self):
        await super().init(
            timeout=g_config.gemini.timeout,
            auto_refresh=g_config.gemini.auto_refresh,
            verbose=g_config.gemini.verbose,
        )

    async def prepare(self, messages: list[Message], tempdir: Path | None = None):
        conversation: list[str] = []
        files: list[Path | str] = []

        for msg in messages:
            if isinstance(msg.content, str):
                # Pure text content
                conversation.append(add_tag(msg.role, msg.content))
            else:
                # Mixed content
                for item in msg.content:
                    if item.type == "text":
                        conversation.append(add_tag(msg.role, item.text or ""))

                    elif item.type == "image_url":
                        # TODO: Use Pydantic to enforce the value checking
                        if not item.image_url:
                            raise ValueError("Image URL cannot be empty")
                        if url := item.image_url.get("url", None):
                            files.append(await save_url_to_tempfile(url, tempdir))
                        else:
                            raise ValueError("Image URL must contain 'url' key")

                    elif item.type == "file":
                        if not item.file:
                            raise ValueError("File cannot be empty")
                        if file_data := item.file.get("file_data", None):
                            filename = item.file.get("filename", "")
                            files.append(await save_file_to_tempfile(file_data, filename, tempdir))
                        else:
                            raise ValueError("File must contain 'file_data' key")

        # Left with the last message as the assistant's response
        conversation.append(add_tag("assistant", "", open=True))
        return conversation, files

    def format_response(self, response: ModelOutput):
        text = ""

        if response.thoughts:
            text += f"<think>{response.thoughts}</think>"

        if response.text:
            text += response.text
        else:
            text += str(response)

        # Fix some escaped characters
        text = text.replace("&lt;", "<").replace("\\<", "<").replace("\\_", "_").replace("\\>", ">")

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
        text = re.sub(pattern, replacer, text)

        # 修复包装的markdown链接
        pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
        return re.sub(pattern, r"\1", text)
