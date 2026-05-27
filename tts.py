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

import os
import struct
from typing import Literal

TtsProvider = Literal["openai", "gemini"]

DEFAULT_GEMINI_TTS_MODEL = "gemini-3.1-flash-live-preview"
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


async def gemini_live_tts(text: str, voice: str = "Kore") -> bytes:
    key_name, api_key = _gemini_api_key()
    if not api_key:
        raise RuntimeError(f"{key_name} is not set")

    from google import genai
    from google.genai import types

    config = types.LiveConnectConfig(
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
    pcm_chunks: list[bytes] = []

    async with client.aio.live.connect(
        model=DEFAULT_GEMINI_TTS_MODEL,
        config=config,
    ) as session:
        await session.send_client_content(
            turns=[types.Content(
                role="user",
                parts=[types.Part.from_text(text=f"Speak exactly this text, nothing else: {text}")]
            )],
            turn_complete=True,
        )
        async for response in session.receive():
            server_content = getattr(response, "server_content", None)
            if server_content is None:
                continue
            model_turn = getattr(server_content, "model_turn", None)
            if model_turn is None:
                continue
            parts = getattr(model_turn, "parts", None)
            if parts is None:
                continue
            for part in parts:
                inline_data = getattr(part, "inline_data", None)
                if inline_data is None:
                    continue
                data = getattr(inline_data, "data", None)
                if data:
                    pcm_chunks.append(data)

    if not pcm_chunks:
        raise RuntimeError("Gemini Live TTS returned no audio data")

    return pcm_to_wav(b"".join(pcm_chunks))
