from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import patch

import policy


IMAGE_DATA_URL = "data:image/jpeg;base64,abc123"


def _completion(content: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


def _raw_decision() -> dict[str, object]:
    return {
        "candidate_found": True,
        "confidence": 0.8,
        "target": {
            "bearing": "center",
            "range": "inside_4m",
            "description": "person under a blue umbrella",
            "free_hand_evidence": "hands are visible and empty",
            "busy_signals": ["none"],
        },
        "safety": {"safe_to_approach": True, "stop_reason": ""},
        "offer": {"drink": True, "photo": True},
        "line": "That blue umbrella is doing serious shade work. Drink, photo, or both?",
        "notes": "available person in range",
    }


def test_missing_gemini_key_returns_default_without_client(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    fetch_policy = policy.FetchPolicy(
        policy.FetchPolicyConfig(
            model="gemini-3.5-flash",
            vision_provider="gemini",
        )
    )

    with patch("policy.OpenAI") as openai_cls:
        decision = fetch_policy.analyze_frame(IMAGE_DATA_URL)

    openai_cls.assert_not_called()
    assert decision["state"] == "search"
    assert decision["notes"] == "GEMINI_API_KEY or GOOGLE_API_KEY is not set"


def test_config_uses_provider_aware_default_model() -> None:
    assert policy.FetchPolicyConfig().model == policy.DEFAULT_OPENAI_VISION_MODEL
    assert (
        policy.FetchPolicyConfig(vision_provider="gemini").model
        == policy.DEFAULT_GEMINI_VISION_MODEL
    )


def test_config_rejects_known_provider_model_mismatch() -> None:
    try:
        policy.FetchPolicyConfig(model="gpt-5-mini", vision_provider="gemini")
    except ValueError as exc:
        assert "appears to be a openai model" in str(exc)
    else:
        raise AssertionError("Expected provider/model mismatch to be rejected")


def test_extract_json_object_ignores_extra_braces_around_response() -> None:
    parsed = policy._extract_json_object(
        'debug {not json} {"candidate_found": false} tail {x}'
    )

    assert parsed == {"candidate_found": False}


def test_invalid_image_url_returns_default_without_client(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    fetch_policy = policy.FetchPolicy()

    with patch("policy.OpenAI") as openai_cls:
        decision = fetch_policy.analyze_frame("not-an-image")

    openai_cls.assert_not_called()
    assert decision["state"] == "search"
    assert decision["notes"] == "Expected an image data URL"


def test_gemini_provider_uses_openai_compatible_endpoint(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")

    fetch_policy = policy.FetchPolicy(
        policy.FetchPolicyConfig(
            model="gemini-3.5-flash",
            vision_provider="gemini",
        )
    )

    with patch("policy.OpenAI") as openai_cls:
        client = openai_cls.return_value
        client.chat.completions.create.return_value = _completion(
            json.dumps(_raw_decision())
        )

        decision = fetch_policy.analyze_frame(IMAGE_DATA_URL)

    openai_cls.assert_called_once_with(
        api_key="gemini-key",
        base_url=policy.GEMINI_OPENAI_BASE_URL,
        timeout=policy.DEFAULT_REQUEST_TIMEOUT_S,
        max_retries=policy.DEFAULT_MAX_RETRIES,
    )
    call_kwargs = client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "gemini-3.5-flash"
    assert call_kwargs["response_format"] == {"type": "json_object"}
    assert call_kwargs["messages"][0]["content"][0]["type"] == "image_url"
    assert decision["state"] == "greet"
    assert decision["line"]


def test_gemini_retries_without_json_mode_if_unsupported(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")

    fetch_policy = policy.FetchPolicy(
        policy.FetchPolicyConfig(
            model="gemini-3.5-flash",
            vision_provider="gemini",
        )
    )

    with patch("policy.OpenAI") as openai_cls:
        client = openai_cls.return_value
        client.chat.completions.create.side_effect = [
            RuntimeError("response_format json_object is not supported"),
            _completion(json.dumps(_raw_decision())),
        ]

        decision = fetch_policy.analyze_frame(IMAGE_DATA_URL)

    calls = client.chat.completions.create.call_args_list
    openai_cls.assert_called_once_with(
        api_key="google-key",
        base_url=policy.GEMINI_OPENAI_BASE_URL,
        timeout=policy.DEFAULT_REQUEST_TIMEOUT_S,
        max_retries=policy.DEFAULT_MAX_RETRIES,
    )
    assert len(calls) == 2
    assert calls[0].kwargs["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1].kwargs
    assert decision["state"] == "greet"


def test_openai_provider_uses_openai_key(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    fetch_policy = policy.FetchPolicy(
        policy.FetchPolicyConfig(model="gpt-5-mini", vision_provider="openai")
    )

    with patch("policy.OpenAI") as openai_cls:
        client = openai_cls.return_value
        client.chat.completions.create.return_value = _completion(
            json.dumps(_raw_decision())
        )

        decision = fetch_policy.analyze_frame(IMAGE_DATA_URL)

    openai_cls.assert_called_once_with(
        api_key="openai-key",
        timeout=policy.DEFAULT_REQUEST_TIMEOUT_S,
        max_retries=policy.DEFAULT_MAX_RETRIES,
    )
    assert decision["state"] == "greet"


def test_client_cache_includes_model(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")

    fetch_policy = policy.FetchPolicy(
        policy.FetchPolicyConfig(model="gpt-5-mini", vision_provider="openai")
    )

    with patch("policy.OpenAI") as openai_cls:
        first_client = openai_cls.return_value
        first_client.chat.completions.create.return_value = _completion(
            json.dumps(_raw_decision())
        )

        fetch_policy.analyze_frame(IMAGE_DATA_URL)
        fetch_policy.config = policy.FetchPolicyConfig(
            model="gpt-5", vision_provider="openai"
        )
        fetch_policy.analyze_frame(IMAGE_DATA_URL)

    assert openai_cls.call_count == 2
