import os
import uuid
import requests
from config.logger import setup_logging
from datetime import datetime
from core.providers.tts.base import TTSProviderBase

TAG = __name__
logger = setup_logging()

class TTSProvider(TTSProviderBase):
    def __init__(self, config, delete_audio_file):
        super().__init__(config, delete_audio_file)
        self.url = config.get("url")
        self.headers = config.get("headers", {})
        self.payload = config.get("payload", {})  # Changed from params to payload
        self.format = config.get("format", "wav")
        self.output_file = config.get("output_dir", "tmp/")

    def generate_filename(self):
        return os.path.join(self.output_file, f"tts-{datetime.now().date()}@{uuid.uuid4().hex}.{self.format}")

    async def text_to_speak(self, text, output_file):
        request_payload = {}
        for k, v in self.payload.items():
            if isinstance(v, str) and "{prompt_text}" in v:
                v = v.replace("{prompt_text}", text)
            request_payload[k] = v

        resp = requests.post(self.url, json=request_payload, headers=self.headers)
        if resp.status_code == 200:
            with open(output_file, "wb") as file:
                file.write(resp.content)
        else:
            logger.bind(tag=TAG).error(f"ElevenlabsTTS: {resp.status_code} - {resp.text}")