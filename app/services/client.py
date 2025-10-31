import asyncio
import re
from pathlib import Path

from gemini_webapi import GeminiClient, ModelOutput
from gemini_webapi.client import ChatSession
from gemini_webapi.constants import Model
from gemini_webapi.exceptions import ModelInvalid
from gemini_webapi.types import Gem

from ..models import Message
from ..utils import g_config
from ..utils.helper import add_tag, save_file_to_tempfile, save_url_to_tempfile

XML_WRAP_HINT = "\nFor any xml block, e.g. tool call, always wrap it with: \n`````xml\n...\n`````\n"


class GeminiClientWrapper(GeminiClient):
    """Gemini client with helper methods."""

    def __init__(self, client_id: str, **kwargs):
        super().__init__(**kwargs)
        self.id = client_id

    async def init(self, **kwargs):
        """
        Inject default configuration values.
        """
        kwargs.setdefault("timeout", g_config.gemini.timeout)
        kwargs.setdefault("auto_refresh", g_config.gemini.auto_refresh)
        kwargs.setdefault("verbose", g_config.gemini.verbose)
        kwargs.setdefault("refresh_interval", g_config.gemini.refresh_interval)
        print(kwargs)
        await super().init(**kwargs)

    async def generate_content(
        self,
        prompt: str,
        files: list[str | Path] | None = None,
        model: Model | str = Model.UNSPECIFIED,
        gem: Gem | str | None = None,
        chat: ChatSession | None = None,
        **kwargs,
    ):
        cnt = 2  # Try 2 times before giving up
        last_exception = None
        while cnt:
            cnt -= 1
            try:
                return await super().generate_content(prompt, files, model, gem, chat, **kwargs)
            except ModelInvalid as e:
                # This is not always caused by model selection. Instead it can be solved by retrying.
                # So we catch it and retry as a workaround.
                await asyncio.sleep(1)
                last_exception = e

        # If retrying failed, re-raise ModelInvalid
        if last_exception:
            raise last_exception

    @staticmethod
    async def process_message(
        message: Message, tempdir: Path | None = None, tagged: bool = True
    ) -> tuple[str, list[Path | str]]:
        """
        Process a single message and return model input.
        """
        model_input = ""
        files: list[Path | str] = []
        if isinstance(message.content, str):
            # Pure text content
            model_input = message.content
        else:
            # Mixed content
            # TODO: Use Pydantic to enforce the value checking
            for item in message.content:
                if item.type == "text":
                    # Append multiple text fragments
                    if item.text:
                        if model_input:
                            model_input += "\n" + item.text
                        else:
                            model_input = item.text

                elif item.type == "image_url":
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

        # This is a workaround for Gemini Web's displaying issues with XML blocks.
        # Add this for tool calling
        if re.search(r"<\s*[^>]+>", model_input):
            hint = XML_WRAP_HINT
        else:
            hint = ""

        # Add role tag if needed
        if model_input:
            if tagged:
                model_input = add_tag(message.role, model_input + hint)
            else:
                model_input += hint

        return model_input, files

    @staticmethod
    async def process_conversation(
        messages: list[Message], tempdir: Path | None = None
    ) -> tuple[str, list[Path | str]]:
        """
        Process the entire conversation and return a formatted string and list of
        files. The last message is assumed to be the assistant's response.
        """
        # Determine once whether we need to wrap messages with role tags: only required
        # if the history already contains assistant/system messages. When every message
        # so far is from the user, we can skip tagging entirely.
        need_tag = any(m.role != "user" for m in messages)

        conversation: list[str] = []
        files: list[Path | str] = []

        for msg in messages:
            input_part, files_part = await GeminiClientWrapper.process_message(
                msg, tempdir, tagged=need_tag
            )
            conversation.append(input_part)
            files.extend(files_part)

        # Append an opening assistant tag only when we used tags above so that Gemini
        # knows where to start its reply.
        if need_tag:
            conversation.append(add_tag("assistant", "", unclose=True))

        return "\n".join(conversation), files

    @staticmethod
    def extract_output(response: ModelOutput, include_thoughts: bool = True) -> str:
        """
        Extract and format the output text from the Gemini response.
        """
        text = ""

        if include_thoughts and response.thoughts:
            text += f"<think>{response.thoughts}</think>\n"

        if response.text:
            text += response.text
        else:
            text += str(response)

        # 保留图片处理在专用方法中，不在此直接输出日志

        # Fix some escaped characters
        text = text.replace("&lt;", "<").replace("\\<", "<").replace("\\_", "_").replace("\\>", ">")

        def simplify_link_target(text_content: str) -> str:
            match_colon_num = re.match(r"([^:]+:\d+)", text_content)
            if match_colon_num:
                return match_colon_num.group(1)
            return text_content

        def replacer(match: re.Match) -> str:
            outer_open_paren = match.group(1)
            display_text = match.group(2)

            new_target_url = simplify_link_target(display_text)
            new_link_segment = f"[`{display_text}`]({new_target_url})"

            if outer_open_paren:
                return f"{outer_open_paren}{new_link_segment})"
            else:
                return new_link_segment

        # Replace Google search links with simplified markdown links
        pattern = r"(\()?\[`([^`]+?)`\]\((https://www.google.com/search\?q=)(.*?)(?<!\\)\)\)*(\))?"
        text = re.sub(pattern, replacer, text)

        # Fix inline code blocks
        pattern = r"`(\[[^\]]+\]\([^\)]+\))`"
        return re.sub(pattern, r"\1", text)

    @staticmethod
    def extract_images(response: ModelOutput) -> list[str]:
        """从 ModelOutput 中提取图片 URL 列表。"""
        urls: list[str] = []
        try:
            images = getattr(response, "images", None)
            if images:
                for image in images:
                    url = getattr(image, "url", None)
                    if isinstance(url, str) and url:
                        urls.append(url)
        except Exception:
            pass
        return urls

    @staticmethod
    def get_image_objects(response: ModelOutput) -> list:
        """从 ModelOutput 中获取图片对象列表（用于保存）。"""
        images = []
        try:
            image_list = getattr(response, "images", None)
            if image_list:
                images = list(image_list)
        except Exception:
            pass
        return images
