from gemini_webapi import GeminiClient
from gemini_webapi.constants import Model
from loguru import logger

from ..utils import g_config
from ..utils.singleton import Singleton


class SingletonGeminiClient(GeminiClient, metaclass=Singleton):
    def __init__(self, **kwargs):
        # TODO: Add proxy support if needed
        super().__init__(
            secure_1psid=g_config.gemini.secure_1psid,
            secure_1psidts=g_config.gemini.secure_1psidts,
            timeout=g_config.gemini.timeout,
            **kwargs,
        )


def map_model_name(openai_model_name: str) -> Model:
    """
    Map OpenAI model name to Gemini model

    Args:
        openai_model_name: OpenAI format model name

    Returns:
        Corresponding Gemini model
    """
    # Print all available models for debugging
    all_models = [m.model_name if hasattr(m, "model_name") else str(m) for m in Model]
    logger.debug(f"Available models: {all_models}")

    # First try to find direct matching model name
    for m in Model:
        model_name = m.model_name if hasattr(m, "model_name") else str(m)
        if openai_model_name.lower() in model_name.lower():
            return m

    # If no match found, use keyword mapping
    model_keywords = {
        "gemini-pro": ["pro", "2.0"],
        "gemini-pro-vision": ["vision", "pro"],
        "gemini-flash": ["flash", "2.0"],
        "gemini-1.5-pro": ["1.5", "pro"],
        "gemini-1.5-flash": ["1.5", "flash"],
    }

    # Match based on keywords
    keywords = model_keywords.get(openai_model_name, ["pro"])  # Default to pro model

    for m in Model:
        model_name = m.model_name if hasattr(m, "model_name") else str(m)
        if all(kw.lower() in model_name.lower() for kw in keywords):
            return m

    # If still not found, return first model
    logger.warning(f"Model {openai_model_name} not found, using default model")
    return next(iter(Model))
