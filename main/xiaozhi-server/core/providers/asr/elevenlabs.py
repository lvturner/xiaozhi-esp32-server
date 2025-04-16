import os
import requests
import uuid
from abc import ABC, abstractmethod
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path
import wave
import opuslib_next

from core.providers.asr.base import ASRProviderBase
from config.logger import setup_logging

TAG = __name__
logger = setup_logging()


class ASRProvider(ASRProviderBase):
    def __init__(self, config: dict, delete_audio_file: bool = True):
        """
        Initialize the ElevenLabs ASR provider.
        Reads configuration from config file.
        """
        self.api_key = config.get("api_key")
        self.model_id = config.get("model_id", "scribe_v1")
        self.base_url = config.get("stt_base_url", "https://api.elevenlabs.io/v1/speech-to-text")
        
        if not self.api_key:
            raise ValueError("ElevenLabs API key not found in configuration")
        
        # Create temp directory if it doesn't exist
        self.output_dir = Path(config.get("temp_directory", "temp/audio"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def save_audio_to_file(self, opus_data: List[bytes], session_id: str) -> str:
        """Decode Opus audio data and save as WAV file"""
        file_name = f"elevenlabs_asr_{session_id}_{uuid.uuid4()}.wav"
        file_path = os.path.join(self.output_dir, file_name)
        
        # Initialize Opus decoder with 16kHz, mono
        decoder = opuslib_next.Decoder(16000, 1)
        pcm_data = []
        
        for opus_packet in opus_data:
            try:
                # Decode with 960 samples (60ms frames at 16kHz)
                pcm_frame = decoder.decode(opus_packet, 960)
                pcm_data.append(pcm_frame)
            except opuslib_next.OpusError as e:
                logger.bind(tag=TAG).error(f"Opus decoding error: {e}", exc_info=True)
        
        # Write to WAV file
        with wave.open(file_path, "wb") as wf:
            wf.setnchannels(1)          # Mono
            wf.setsampwidth(2)          # 16-bit (2 bytes)
            wf.setframerate(16000)      # 16kHz sample rate
            wf.writeframes(b"".join(pcm_data))
        
        return file_path

    async def speech_to_text(
        self,
        opus_data: List[bytes],
        session_id: str,
        language_code: Optional[str] = None,
        tag_audio_events: bool = True,
        num_speakers: Optional[int] = None,
        timestamps_granularity: str = "word",
        diarize: bool = False,
        enable_logging: bool = True
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Convert speech data to text using ElevenLabs API (synchronous version)
        
        Args:
            opus_data: List of Opus audio packets
            session_id: Unique session identifier
            language_code: ISO-639-1 or ISO-639-3 language code (optional)
            tag_audio_events: Whether to tag audio events (default: True)
            num_speakers: Maximum number of speakers (1-32, optional)
            timestamps_granularity: 'word' or 'character' (default: 'word')
            diarize: Whether to annotate speakers (default: False)
            enable_logging: Whether to enable logging (default: True)
            
        Returns:
            Tuple of (transcribed_text, error_message)
        """
        # First save the audio to a file
        audio_file_path = self.save_audio_to_file(opus_data, session_id)
        
        if not os.path.exists(audio_file_path):
            logger.bind(tag=TAG).error(f"Audio file not created: {audio_file_path}")
            return None, None
        
        headers = {
            "xi-api-key": self.api_key,
        }
        
        # Prepare form data
        data = {
            "model_id": self.model_id,
            "enable_logging": str(enable_logging).lower(),
            "tag_audio_events": str(tag_audio_events).lower(),
            "timestamps_granularity": timestamps_granularity,
            "diarize": str(diarize).lower(),
            "file_format": "pcm_s16le_16"  # Since we're providing 16kHz 16-bit PCM
        }
        
        if language_code:
            data["language_code"] = language_code
        if num_speakers:
            data["num_speakers"] = str(num_speakers)
        
        try:
            with open(audio_file_path, 'rb') as audio_file:
                # Note: Using 'data' for form fields and 'files' for the file
                response = requests.post(
                    self.base_url,
                    headers=headers,
                    data=data,
                    files={'file': audio_file}
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "").strip()
                    return text, None
                else:
                    error = response.text
                    logger.bind(tag=TAG).error(f"ElevenLabs API error: {error}")
                    return None, error
        except Exception as e:
            logger.bind(tag=TAG).error(f"Error in ElevenLabs ASR: {str(e)}", exc_info=True)
            return None, str(e)
        finally:
            # Clean up the temporary file
            try:
                os.remove(audio_file_path)
            except Exception as e:
                logger.bind(tag=TAG).warning(f"Could not remove temp file {audio_file_path}: {e}")