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

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from typing import Any, Literal

from openai import OpenAI

ApproachState = Literal["search", "approach", "greet", "wait_for_bottle", "photo_ready", "skip"]
InteractionPhase = Literal["find_guest", "confirm_bottle"]
Bearing = Literal["left", "center", "right", "unknown"]
RangeEstimate = Literal["far", "medium", "near", "inside_4m", "inside_1m", "unknown"]
VisionProvider = Literal["openai", "gemini"]

GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"
DEFAULT_OPENAI_VISION_MODEL = "gpt-5-mini"
DEFAULT_GEMINI_VISION_MODEL = "gemini-3.5-flash"
DEFAULT_VISION_MODEL_BY_PROVIDER: dict[VisionProvider, str] = {
    "openai": DEFAULT_OPENAI_VISION_MODEL,
    "gemini": DEFAULT_GEMINI_VISION_MODEL,
}
DEFAULT_REQUEST_TIMEOUT_S = 30.0
DEFAULT_MAX_RETRIES = 2


def _validate_vision_provider(provider: str) -> VisionProvider:
    if provider == "openai" or provider == "gemini":
        return provider
    raise ValueError(f"Unsupported vision provider: {provider!r}")


def default_model_for_provider(provider: VisionProvider) -> str:
    return DEFAULT_VISION_MODEL_BY_PROVIDER[_validate_vision_provider(provider)]


def _known_provider_for_model(model: str) -> VisionProvider | None:
    normalized = model.lower().strip()
    if normalized.startswith(("gemini-", "models/gemini-")):
        return "gemini"
    if normalized.startswith("gpt-") or re.match(r"^o\d(?:-|$)", normalized):
        return "openai"
    return None


def _validate_model_for_provider(provider: VisionProvider, model: str) -> None:
    model_provider = _known_provider_for_model(model)
    if model_provider is not None and model_provider != provider:
        raise ValueError(
            f"Model {model!r} appears to be a {model_provider} model, "
            f"but vision_provider is {provider!r}"
        )


@dataclass(frozen=True)
class FetchPolicyConfig:
    model: str | None = None
    vision_provider: VisionProvider = "openai"
    max_line_chars: int = 180
    request_timeout_s: float = DEFAULT_REQUEST_TIMEOUT_S
    max_retries: int = DEFAULT_MAX_RETRIES

    def __post_init__(self) -> None:
        provider = _validate_vision_provider(self.vision_provider)
        model = (
            default_model_for_provider(provider)
            if self.model is None
            else self.model
        )
        if not isinstance(model, str):
            raise ValueError("Vision model must be a string")
        model = model.strip()
        if not model:
            raise ValueError("Vision model must be a non-empty string")
        _validate_model_for_provider(provider, model)
        if (
            not isinstance(self.max_line_chars, int)
            or isinstance(self.max_line_chars, bool)
            or self.max_line_chars < 1
        ):
            raise ValueError("max_line_chars must be positive")
        if (
            not isinstance(self.request_timeout_s, (int, float))
            or isinstance(self.request_timeout_s, bool)
            or self.request_timeout_s <= 0
        ):
            raise ValueError("request_timeout_s must be positive")
        if (
            not isinstance(self.max_retries, int)
            or isinstance(self.max_retries, bool)
            or self.max_retries < 0
        ):
            raise ValueError("max_retries cannot be negative")

        object.__setattr__(self, "vision_provider", provider)
        object.__setattr__(self, "model", model)


@dataclass(frozen=True)
class _ClientCacheKey:
    api_key: str
    provider: VisionProvider
    model: str
    request_timeout_s: float
    max_retries: int


def _default_decision(reason: str) -> dict[str, Any]:
    return {
        "type": "decision",
        "state": "search",
        "candidate_found": False,
        "confidence": 0.0,
        "target": {
            "bearing": "unknown",
            "range": "unknown",
            "description": "",
            "free_hand_evidence": "",
            "busy_signals": [],
        },
        "safety": {
            "safe_to_approach": False,
            "stop_reason": reason,
        },
        "offer": {
            "drink": True,
            "photo": True,
        },
        "action": "search",
        "photo_ready": False,
        "bottle_visible": False,
        "framing": {
            "person_visible": False,
            "bottle_visible": False,
            "well_framed": False,
            "notes": "",
        },
        "line": "",
        "simulated_cmd_vel": {
            "linear_x": 0.0,
            "angular_z": 0.35,
            "duration_s": 0.8,
        },
        "notes": reason,
    }


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped)
        stripped = re.sub(r"\s*```$", "", stripped)

    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError as original_exc:
        decoder = json.JSONDecoder()
        parsed = None
        for index, char in enumerate(stripped):
            if char != "{":
                continue
            try:
                candidate, _ = decoder.raw_decode(stripped[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(candidate, dict):
                parsed = candidate
                break
        if parsed is None:
            raise original_exc

    if not isinstance(parsed, dict):
        raise ValueError("Vision model response was not a JSON object")
    return parsed


def _api_key_for_provider(provider: VisionProvider) -> tuple[str, str | None]:
    provider = _validate_vision_provider(provider)
    if provider == "openai":
        return "OPENAI_API_KEY", os.getenv("OPENAI_API_KEY")
    if provider == "gemini":
        return (
            "GEMINI_API_KEY or GOOGLE_API_KEY",
            os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        )
    raise ValueError(f"Unsupported vision provider: {provider!r}")


def _uses_unsupported_response_format(exc: Exception) -> bool:
    message = str(exc).lower()
    return "response_format" in message or "json_object" in message


def _as_bearing(value: Any) -> Bearing:
    if value in {"left", "center", "right", "unknown"}:
        return value
    return "unknown"


def _as_range(value: Any) -> RangeEstimate:
    if value in {"far", "medium", "near", "inside_4m", "inside_1m", "unknown"}:
        return value
    return "unknown"


def _clean_spoken_line(line: str) -> str:
    line = re.sub(
        r"^(?:find[_-]?guest|confirm[_-]?bottle)\b[\s:|\-\u2013\u2014]*",
        "",
        line.strip(),
        flags=re.IGNORECASE,
    )
    line = re.sub(r"\*[^*]{0,80}\*\s*", "", line)
    return re.sub(r"\s+", " ", line).strip()


def _truncate_spoken_line(line: str, max_chars: int) -> str:
    if len(line) <= max_chars:
        return line
    return (
        re.sub(
            r"\b(?:a|an|and|or|the|to|for|with|of|in|on|at|by)\.?$",
            "",
            line[:max_chars].rsplit(" ", 1)[0].rstrip(".,;:"),
            flags=re.IGNORECASE,
        ).rstrip(" ,;:")
        + "."
    )


def _mentions_coke_container(lower_line: str) -> bool:
    if re.search(r"\b(coke|cola|soda|bottle|drink)\b", lower_line):
        return True
    return (
        re.search(r"\b(?:coke|cola|soda)\s+can\b", lower_line) is not None
        or re.search(r"\bcan\s+of\s+(?:coke|cola|soda)\b", lower_line) is not None
        or re.search(r"\b(?:the|a|your|that)\s+can\b", lower_line) is not None
    )


def _ensure_script_line(
    line: str,
    state: ApproachState,
    interaction_phase: InteractionPhase,
) -> str:
    if state not in {"greet", "photo_ready", "wait_for_bottle"}:
        return ""

    cleaned = line.strip()
    lower = cleaned.lower()

    if state == "greet":
        has_coke = _mentions_coke_container(lower)
        has_back = re.search(r"\b(back|bag|pack)\b", lower) is not None
        if has_coke and has_back:
            return cleaned
        suffix = "Grab a Coke from my back first, then I will take your instant photo."
    elif interaction_phase == "confirm_bottle" and state == "wait_for_bottle":
        has_coke = _mentions_coke_container(lower)
        has_frame = re.search(r"\b(frame|camera|photo|front|center|hold|show)\b", lower) is not None
        if has_coke and has_frame:
            return cleaned
        suffix = "Hold the Coke can out front and center yourself in the camera frame."
    else:
        has_coke = _mentions_coke_container(lower)
        has_cue = re.search(r"\b(cheers|cheese|camera|photo|frame)\b", lower) is not None
        if has_coke and has_cue:
            return cleaned
        suffix = "Hold the Coke up front and center. Cheers."

    if not cleaned:
        return suffix
    return f"{cleaned.rstrip('.!?')}. {suffix}"


def _cmd_for_target(bearing: Bearing, range_estimate: RangeEstimate) -> dict[str, float]:
    if range_estimate in {"inside_4m", "inside_1m"}:
        return {"linear_x": 0.0, "angular_z": 0.0, "duration_s": 0.0}

    angular_z = 0.0
    if bearing == "left":
        angular_z = 0.28
    elif bearing == "right":
        angular_z = -0.28
    elif bearing == "unknown":
        angular_z = 0.35

    linear_x = 0.0
    duration_s = 0.8
    if bearing in {"left", "center", "right"}:
        linear_x = 0.22 if range_estimate == "near" else 0.32
        duration_s = 0.9 if range_estimate == "near" else 1.2

    return {"linear_x": linear_x, "angular_z": angular_z, "duration_s": duration_s}


def _normalize_decision(
    raw: dict[str, Any],
    config: FetchPolicyConfig,
    interaction_phase: InteractionPhase = "find_guest",
) -> dict[str, Any]:
    candidate_found = bool(raw.get("candidate_found"))
    confidence = float(raw.get("confidence") or 0.0)
    target = raw.get("target") if isinstance(raw.get("target"), dict) else {}
    safety = raw.get("safety") if isinstance(raw.get("safety"), dict) else {}
    offer = raw.get("offer") if isinstance(raw.get("offer"), dict) else {}
    framing = raw.get("framing") if isinstance(raw.get("framing"), dict) else {}

    bearing = _as_bearing(target.get("bearing"))
    range_estimate = _as_range(target.get("range"))
    safe_to_approach = bool(safety.get("safe_to_approach"))
    photo_ready = bool(raw.get("photo_ready") or framing.get("well_framed"))
    bottle_visible = bool(raw.get("bottle_visible") or framing.get("bottle_visible"))
    line = _clean_spoken_line(str(raw.get("line") or ""))

    if interaction_phase == "confirm_bottle":
        if photo_ready:
            state: ApproachState = "photo_ready"
        elif candidate_found:
            state = "wait_for_bottle"
        else:
            state = "search"
    elif not candidate_found:
        state: ApproachState = "search"
    elif not safe_to_approach:
        state = "skip"
    elif range_estimate in {"inside_4m", "inside_1m"}:
        state = "greet"
    else:
        state = "approach"

    if state not in {"greet", "photo_ready", "wait_for_bottle"}:
        line = ""
    else:
        line = _truncate_spoken_line(
            _ensure_script_line(line, state, interaction_phase),
            config.max_line_chars,
        )

    cmd = (
        _cmd_for_target(bearing, range_estimate)
        if state == "approach"
        else _default_decision("")["simulated_cmd_vel"]
    )
    if state in {"greet", "wait_for_bottle", "photo_ready"}:
        cmd = {"linear_x": 0.0, "angular_z": 0.0, "duration_s": 0.0}

    action_by_state = {
        "search": "search",
        "approach": "approach",
        "greet": "wave_offer",
        "wait_for_bottle": "coach_photo",
        "photo_ready": "take_photo_dance",
        "skip": "skip",
    }

    return {
        "type": "decision",
        "state": state,
        "action": str(raw.get("action") or action_by_state[state]),
        "candidate_found": candidate_found,
        "confidence": max(0.0, min(1.0, confidence)),
        "target": {
            "bearing": bearing,
            "range": range_estimate,
            "description": str(target.get("description") or ""),
            "free_hand_evidence": str(target.get("free_hand_evidence") or ""),
            "lying_evidence": str(target.get("lying_evidence") or ""),
            "happy_evidence": str(target.get("happy_evidence") or ""),
            "busy_signals": list(target.get("busy_signals") or []),
        },
        "safety": {
            "safe_to_approach": safe_to_approach,
            "stop_reason": str(safety.get("stop_reason") or ""),
        },
        "offer": {
            "drink": bool(offer.get("drink", True)),
            "photo": bool(offer.get("photo", True)),
        },
        "photo_ready": photo_ready,
        "bottle_visible": bottle_visible,
        "framing": {
            "person_visible": bool(framing.get("person_visible", candidate_found)),
            "bottle_visible": bottle_visible,
            "well_framed": photo_ready,
            "notes": str(framing.get("notes") or ""),
        },
        "line": line,
        "simulated_cmd_vel": cmd,
        "notes": str(raw.get("notes") or ""),
    }


class FetchPolicy:
    """Vision policy for the Fetch interaction prototype."""

    def __init__(self, config: FetchPolicyConfig | None = None) -> None:
        self.config = config or FetchPolicyConfig()
        self._client: OpenAI | None = None
        self._client_cache_key: _ClientCacheKey | None = None

    def _get_client(self, api_key: str) -> OpenAI:
        cache_key = _ClientCacheKey(
            api_key=api_key,
            provider=self.config.vision_provider,
            model=self.config.model,
            request_timeout_s=self.config.request_timeout_s,
            max_retries=self.config.max_retries,
        )
        if self._client is not None and self._client_cache_key == cache_key:
            return self._client

        client_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": self.config.request_timeout_s,
            "max_retries": self.config.max_retries,
        }
        if self.config.vision_provider == "gemini":
            client_kwargs["base_url"] = GEMINI_OPENAI_BASE_URL

        self._client = OpenAI(**client_kwargs)
        self._client_cache_key = cache_key
        return self._client

    def _create_completion(
        self,
        client: OpenAI,
        messages: list[dict[str, Any]],
        *,
        use_json_response_format: bool,
    ) -> Any:
        kwargs: dict[str, Any] = {
            "model": self.config.model,
            "messages": messages,
        }
        if use_json_response_format:
            kwargs["response_format"] = {"type": "json_object"}
        return client.chat.completions.create(**kwargs)

    def analyze_frame(
        self,
        image_data_url: str,
        depth_hint: dict[str, Any] | None = None,
        interaction_phase: InteractionPhase = "find_guest",
    ) -> dict[str, Any]:
        api_key_name, api_key = _api_key_for_provider(self.config.vision_provider)
        if not api_key:
            return _default_decision(f"{api_key_name} is not set")

        if not image_data_url.startswith("data:image/"):
            return _default_decision("Expected an image data URL")

        depth_note = "No depth estimate was provided."
        if depth_hint:
            depth_note = f"Depth hint from client: {json.dumps(depth_hint, sort_keys=True)}"

        if interaction_phase == "confirm_bottle":
            goal = """
Current phase: confirm_bottle.
- The dog has already waved and told the person to take a Coke can from the dog's back.
- Look for the same or primary person holding a Coke can, Coke bottle, soda can, or clearly bottle-shaped drink from the dog's back.
- The person and Coke must both be visible and well framed for a photo. Prefer face/upper body plus the Coke held out front; reject if the Coke is cropped out, hidden, or too blurry.
- If ready, set photo_ready true, action "take_photo_dance", and line to a short photographer cue that tells them to hold the Coke up/front and says "cheers", "cheese", or a funny camera phrase.
- If the person is visible but the Coke is missing or framing is bad, set candidate_found true, photo_ready false, action "coach_photo", and line to one clear instruction: take/hold the Coke can, put it in front, and move/center themselves for the camera.
"""
        else:
            goal = """
Current phase: find_guest.
- Find the single best visible target for a Coca-Cola marketing robot dog giving out free Cokes and taking instant photos.
- Pick anyone who looks chill and likely to enjoy the bit: visibly thirsty, curious, amused, playful, social, looking toward the dog/camera, or otherwise like they would be in for a good laugh and a photo.
- Do not require the person to be lying down, reclining, alone, empty-handed, or fully idle.
- Phone, book, laptop, food, or existing drink are not automatic rejects. Treat them as weak busy signals only when the person looks engrossed, unavailable, or likely to be annoyed.
- Prefer open posture, visible face/upper body, a clear path, and enough room to stop safely.
- The dog should approach only if the path looks safe, then stop within a few meters.
- Generate the greeting only when the target is inside 4 meters.
- The greeting must wave, make a highly personalized joke from visible non-sensitive appearance/context, clearly tell them to take a Coke can from the dog's back first, then tell them they get an instant photo.
- Be explicit about the sequence: take Coke from my back, then pose for the photo with the Coke.
"""

        prompt = f"""
You are the behavior policy for a small robot dog Fetch.

Goal:
{goal}

Safety and privacy:
- Do not identify people or infer sensitive traits.
- Do not comment on race, ethnicity, gender, age, disability, body size, attractiveness, or medical state.
- Avoid insults and body-shaming. Keep humor based on visible, non-sensitive details: setting, posture, lighting, colors, bags, objects nearby, or what is happening in the scene.
- The line must be specific to this frame. Mention at least one concrete visible detail from the person or scene. Do not write generic lines like "Want a drink or photo?" or "You look thirsty."
- The line is spoken aloud. Do not include markdown, asterisks, phase labels, narration, or stage directions such as waving; the robot wave and photo sound are handled separately.
- Keep the line friendly and directive: the person should always know the next physical step.
- If the path looks blocked or the target is too close to another person or object, mark safe_to_approach false.

Return only JSON with this shape:
{{
  "candidate_found": true,
  "confidence": 0.0,
  "target": {{
    "bearing": "left|center|right|unknown",
    "range": "far|medium|near|inside_4m|inside_1m|unknown",
    "description": "short visible description",
    "free_hand_evidence": "why they seem able or likely to accept a Coke/photo",
    "lying_evidence": "visible posture or stance; do not require reclining",
    "happy_evidence": "why they seem receptive, amused, thirsty, curious, or social",
    "busy_signals": ["phone", "drink", "book", "work", "none"]
  }},
  "safety": {{
    "safe_to_approach": true,
    "stop_reason": ""
  }},
  "offer": {{
    "drink": true,
    "photo": true
  }},
  "action": "search|approach|wave_offer|coach_photo|take_photo_dance|skip",
  "photo_ready": false,
  "bottle_visible": false,
  "framing": {{
    "person_visible": true,
    "bottle_visible": false,
    "well_framed": false,
    "notes": "short framing notes"
  }},
  "line": "one short spoken dog line; it must reference visible context and include no labels or stage directions",
  "notes": "short reasoning"
}}

Range rule:
- Use "inside_4m" when the camera is already close enough for greeting and a snapshot.
- Use "inside_1m" only when the person is clearly within 1 meter.
- Use "near" when the person is close but not clearly within 4 meters.
- If the depth hint center_median_m, center_p10_m, frame_median_m, or frame_p10_m is 4.0 or less and a visible receptive target is in that region, prefer "inside_4m".

{depth_note}
""".strip()

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        client = self._get_client(api_key)
        try:
            response = self._create_completion(
                client,
                messages,
                use_json_response_format=True,
            )
        except Exception as exc:
            if (
                self.config.vision_provider != "gemini"
                or not _uses_unsupported_response_format(exc)
            ):
                raise
            response = self._create_completion(
                client,
                messages,
                use_json_response_format=False,
            )
        content = response.choices[0].message.content or "{}"
        parsed = _extract_json_object(content)
        return _normalize_decision(parsed, self.config, interaction_phase)
