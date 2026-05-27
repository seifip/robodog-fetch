# Copyright 2026 Dimensional Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-to-speech providers for the Fetch prototype."""

from __future__ import annotations

import base64
import os
import re
import struct
from typing import Any, Literal

TtsProvider = Literal["openai", "gemini"]

DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-tts-preview"
GEMINI_TTS_SAMPLE_RATE = 24000

VOICE_MAP_OPENAI_TO_GEMINI: dict[str, str] = {
    "alloy": "Kore",
    "echo": "Charon",
    "fable": "Puck",
    "onyx": "Orus",
    "nova": "Aoede",
    "shimmer": "Led",
}

GEMINI_PREBUILT_VOICES = frozenset({
    "Achernar", "Achird", "Algieba", "Algenib", "Aoede", "Autonoe",
    "Capella", "Charon", "Despina", "Erinome", "Fenrir", "Kore",
    "Laomedeia", "Led", "Oberon", "Orus", "Pollux", "Puck",
    "Rasalgethi", "Sadachbia", "Sadaltager", "Schedar", "Sulafat",
    "Titania", "Umbriel", "Zephyr", "Zubenelgenubi",
})


def map_voice(voice: str, tts_provider: TtsProvider) -> str:
    if tts_provider == "gemini":
        if voice in GEMINI_PREBUILT_VOICES:
            return voice
        return VOICE_MAP_OPENAI_TO_GEMINI.get(voice, "Kore")
    return voice


def pcm_to_wav(
    pcm: bytes,
    sample_rate: int = GEMINI_TTS_SAMPLE_RATE,
    channels: int = 1,
    sample_width: int = 2,
) -> bytes:
    data_size = len(pcm)
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + data_size,
        b"WAVE",
        b"fmt ",
        16,
        1,
        channels,
        sample_rate,
        sample_rate * channels * sample_width,
        channels * sample_width,
        sample_width * 8,
        b"data",
        data_size,
    )
    return header + pcm


def _gemini_api_key() -> tuple[str, str | None]:
    return (
        "GEMINI_API_KEY or GOOGLE_API_KEY",
        os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
    )


def _sample_rate_from_mime_type(mime_type: str | None) -> int:
    if not mime_type:
        return GEMINI_TTS_SAMPLE_RATE
    match = re.search(r"(?:^|[;\s])rate=(\d+)", mime_type)
    if match is None:
        return GEMINI_TTS_SAMPLE_RATE
    return int(match.group(1))


def _field(value: Any, name: str) -> Any:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _inline_audio_bytes(inline_data: Any) -> bytes | None:
    data = _field(inline_data, "data")
    if data is None:
        return None
    if isinstance(data, bytes):
        return data
    if isinstance(data, bytearray):
        return bytes(data)
    if isinstance(data, memoryview):
        return data.tobytes()
    if isinstance(data, str):
        try:
            return base64.b64decode(data, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise RuntimeError("Gemini TTS returned invalid base64 audio data") from exc
    return None


def _extract_gemini_tts_audio(response: Any) -> tuple[bytes, str | None]:
    chunks: list[bytes] = []
    mime_type: str | None = None

    for candidate in _field(response, "candidates") or []:
        content = _field(candidate, "content")
        for part in _field(content, "parts") or []:
            inline_data = _field(part, "inline_data")
            if inline_data is None:
                continue
            audio = _inline_audio_bytes(inline_data)
            if not audio:
                continue
            chunks.append(audio)
            mime_type = mime_type or _field(inline_data, "mime_type")

    if not chunks:
        for part in _field(response, "parts") or []:
            inline_data = _field(part, "inline_data")
            if inline_data is None:
                continue
            audio = _inline_audio_bytes(inline_data)
            if not audio:
                continue
            chunks.append(audio)
            mime_type = mime_type or _field(inline_data, "mime_type")

    if not chunks:
        raise RuntimeError("Gemini TTS returned no audio data")

    return b"".join(chunks), mime_type


async def gemini_tts(text: str, voice: str = "Kore") -> bytes:
    text = text.strip()
    if not text:
        raise RuntimeError("Gemini TTS text must be non-empty")

    key_name, api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError(f"{key_name} is not set")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "google-genai is required for Gemini TTS; install it with `pip install google-genai`"
        ) from exc

    config = types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                    voice_name=voice,
                )
            )
        ),
    )

    client = genai.Client(api_key=api_key)
    response = await client.aio.models.generate_content(
        model=DEFAULT_GEMINI_TTS_MODEL,
        contents=(
            "Speak exactly this Fetch robot dog line, with warm playful energy. "
            f"Do not add words: {text}"
        ),
        config=config,
    )
    audio, mime_type = _extract_gemini_tts_audio(response)
    if audio.startswith(b"RIFF") or mime_type in {"audio/wav", "audio/wave", "audio/x-wav"}:
        return audio

    return pcm_to_wav(audio, sample_rate=_sample_rate_from_mime_type(mime_type))


gemini_live_tts = gemini_tts
