from __future__ import annotations

import asyncio
import base64
import struct
import types as builtin_types
from types import SimpleNamespace
from unittest.mock import patch

import pytest

import tts


class _MockModels:
    def __init__(self, response):
        self.response = response
        self.calls = []

    async def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        return self.response


def _make_audio_response(pcm_bytes, mime_type="audio/L16;codec=pcm;rate=24000"):
    inline_data = SimpleNamespace(data=pcm_bytes, mime_type=mime_type)
    part = SimpleNamespace(inline_data=inline_data)
    content = SimpleNamespace(parts=[part])
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def _make_response_from_parts(parts):
    content = SimpleNamespace(parts=parts)
    candidate = SimpleNamespace(content=content)
    return SimpleNamespace(candidates=[candidate])


def test_map_voice_openai_passthrough() -> None:
    assert tts.map_voice("echo", "openai") == "echo"
    assert tts.map_voice("alloy", "openai") == "alloy"


def test_map_voice_gemini_known_openai_name() -> None:
    assert tts.map_voice("echo", "gemini") == "Charon"
    assert tts.map_voice("alloy", "gemini") == "Kore"
    assert tts.map_voice("fable", "gemini") == "Puck"
    assert tts.map_voice("onyx", "gemini") == "Orus"
    assert tts.map_voice("nova", "gemini") == "Aoede"
    assert tts.map_voice("shimmer", "gemini") == "Led"


def test_map_voice_gemini_native_name_passthrough() -> None:
    assert tts.map_voice("Kore", "gemini") == "Kore"
    assert tts.map_voice("Charon", "gemini") == "Charon"
    assert tts.map_voice("Puck", "gemini") == "Puck"


def test_map_voice_gemini_unknown_defaults_to_kore() -> None:
    assert tts.map_voice("unknown_voice", "gemini") == "Kore"


def test_pcm_to_wav_header_structure() -> None:
    pcm = b"\x00\x00" * 100
    wav = tts.pcm_to_wav(pcm)

    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert wav[12:16] == b"fmt "
    assert wav[36:40] == b"data"

    riff_size = struct.unpack_from("<I", wav, 4)[0]
    assert riff_size == 36 + len(pcm)

    data_size = struct.unpack_from("<I", wav, 40)[0]
    assert data_size == len(pcm)

    fmt_chunk_size = struct.unpack_from("<I", wav, 16)[0]
    assert fmt_chunk_size == 16

    audio_format = struct.unpack_from("<H", wav, 20)[0]
    assert audio_format == 1

    num_channels = struct.unpack_from("<H", wav, 22)[0]
    assert num_channels == 1

    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    assert sample_rate == tts.GEMINI_TTS_SAMPLE_RATE

    bits_per_sample = struct.unpack_from("<H", wav, 34)[0]
    assert bits_per_sample == 16

    assert wav[44:] == pcm


def test_pcm_to_wav_custom_params() -> None:
    pcm = b"\xff\x00" * 50
    wav = tts.pcm_to_wav(pcm, sample_rate=48000, channels=2, sample_width=2)

    sample_rate = struct.unpack_from("<I", wav, 24)[0]
    assert sample_rate == 48000

    num_channels = struct.unpack_from("<H", wav, 22)[0]
    assert num_channels == 2

    byte_rate = struct.unpack_from("<I", wav, 28)[0]
    assert byte_rate == 48000 * 2 * 2


def test_gemini_tts_missing_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    with pytest.raises(RuntimeError, match="GEMINI_API_KEY or GOOGLE_API_KEY is not set"):
        asyncio.get_event_loop().run_until_complete(
            tts.gemini_tts("hello"),
        )


def _patch_genai(mock_models):
    mock_client = SimpleNamespace(
        aio=SimpleNamespace(
            models=mock_models,
        ),
    )

    mock_genai_module = builtin_types.ModuleType("google.genai")
    mock_genai_module.Client = lambda **kw: mock_client

    mock_types_module = builtin_types.ModuleType("google.genai.types")
    mock_types_module.GenerateContentConfig = lambda **kw: SimpleNamespace(**kw)
    mock_types_module.SpeechConfig = lambda **kw: None
    mock_types_module.VoiceConfig = lambda **kw: None
    mock_types_module.PrebuiltVoiceConfig = lambda **kw: None

    mock_google = builtin_types.ModuleType("google")
    mock_google.genai = mock_genai_module

    return patch.dict("sys.modules", {
        "google": mock_google,
        "google.genai": mock_genai_module,
        "google.genai.types": mock_types_module,
    })


def test_gemini_tts_returns_wav(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    fake_pcm = b"\xab\xcd" * 500
    mock_models = _MockModels(_make_audio_response(fake_pcm))

    with _patch_genai(mock_models):
        result = asyncio.get_event_loop().run_until_complete(
            tts.gemini_tts("Hello beach!", voice="Kore"),
        )

    assert result[:4] == b"RIFF"
    assert result[8:12] == b"WAVE"
    data_size = struct.unpack_from("<I", result, 40)[0]
    assert data_size == len(fake_pcm)
    assert mock_models.calls
    assert mock_models.calls[0]["model"] == tts.DEFAULT_GEMINI_TTS_MODEL
    assert "Hello beach!" in mock_models.calls[0]["contents"]


def test_gemini_tts_no_audio_raises(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    mock_models = _MockModels(SimpleNamespace(candidates=[]))

    with _patch_genai(mock_models):
        with pytest.raises(RuntimeError, match="no audio data"):
            asyncio.get_event_loop().run_until_complete(
                tts.gemini_tts("Hello!"),
            )


def test_gemini_tts_multi_part_audio(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    chunk1 = b"\x01\x02" * 100
    chunk2 = b"\x03\x04" * 100
    parts = [
        SimpleNamespace(
            inline_data=SimpleNamespace(
                data=chunk1,
                mime_type="audio/L16;codec=pcm;rate=24000",
            )
        ),
        SimpleNamespace(
            inline_data=SimpleNamespace(
                data=chunk2,
                mime_type="audio/L16;codec=pcm;rate=24000",
            )
        ),
    ]
    mock_models = _MockModels(_make_response_from_parts(parts))

    with _patch_genai(mock_models):
        result = asyncio.get_event_loop().run_until_complete(
            tts.gemini_tts("Hello!"),
        )

    assert result[:4] == b"RIFF"
    data_size = struct.unpack_from("<I", result, 40)[0]
    assert data_size == len(chunk1) + len(chunk2)
    assert result[44:] == chunk1 + chunk2


def test_gemini_tts_preserves_wav_response(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    wav = b"RIFF\x00\x00\x00\x00WAVEfake"
    mock_models = _MockModels(_make_audio_response(wav, mime_type="audio/wav"))

    with _patch_genai(mock_models):
        result = asyncio.get_event_loop().run_until_complete(
            tts.gemini_tts("Hello!"),
        )

    assert result == wav


def test_gemini_tts_handles_dict_response_and_sample_rate(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    pcm = b"\x05\x06" * 10
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inline_data": {
                                "data": base64.b64encode(pcm).decode("ascii"),
                                "mime_type": "audio/L16;codec=pcm;rate=16000",
                            }
                        }
                    ]
                }
            }
        ]
    }
    mock_models = _MockModels(response)

    with _patch_genai(mock_models):
        result = asyncio.get_event_loop().run_until_complete(
            tts.gemini_tts("Hello!"),
        )

    assert struct.unpack_from("<I", result, 24)[0] == 16000
    assert result[44:] == pcm
