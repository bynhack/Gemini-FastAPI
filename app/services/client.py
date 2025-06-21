from gemini_webapi import GeminiClient

from ..utils import g_config
from ..utils.singleton import Singleton


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
