from __future__ import annotations

from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import AsyncMock, patch
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

import policy
import tts


def _package(name: str) -> ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = ModuleType(name)
        sys.modules[name] = module
    module.__path__ = []  # type: ignore[attr-defined]
    return module


def _module(name: str, **attrs: object) -> ModuleType:
    module = ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    sys.modules[name] = module
    return module


class _FastAPIServer:
    def __init__(self, **_: object) -> None:
        self.app = FastAPI()

    def run(self, **_: object) -> None:
        return None


class _Record3DSource:
    def __init__(self, *_: object, **__: object) -> None:
        return None


class _UnitreeWebRTCConnection:
    def __init__(self, *_: object, **__: object) -> None:
        return None

    def stop(self) -> None:
        return None


class _Image:
    def to_base64(self, *_: object, **__: object) -> str:
        return ""


class _PointCloud2:
    def as_numpy(self) -> tuple[object, object]:
        return ([], None)


def _setup_logger() -> SimpleNamespace:
    return SimpleNamespace(
        info=lambda *_args, **_kwargs: None,
        exception=lambda *_args, **_kwargs: None,
        warning=lambda *_args, **_kwargs: None,
        error=lambda *_args, **_kwargs: None,
        debug=lambda *_args, **_kwargs: None,
    )


for package in [
    "dimos",
    "dimos.experimental",
    "dimos.experimental.fetch",
    "dimos.msgs",
    "dimos.msgs.geometry_msgs",
    "dimos.msgs.sensor_msgs",
    "dimos.robot",
    "dimos.robot.unitree",
    "dimos.utils",
    "dimos.web",
    "dimos.web.dimos_interface",
    "dimos.web.dimos_interface.api",
]:
    _package(package)

sys.modules["dimos.experimental.fetch.policy"] = policy
sys.modules["dimos.experimental.fetch.tts"] = tts
_module(
    "cv2",
    COLORMAP_TURBO=0,
    FONT_HERSHEY_SIMPLEX=0,
    IMWRITE_JPEG_QUALITY=1,
    applyColorMap=lambda values, _map: values,
    circle=lambda *_args, **_kwargs: None,
    imencode=lambda *_args, **_kwargs: (True, SimpleNamespace(tobytes=lambda: b"jpg")),
    line=lambda *_args, **_kwargs: None,
    putText=lambda *_args, **_kwargs: None,
)
_module("dimos.experimental.fetch.record3d_source", Record3DSource=_Record3DSource)
_module(
    "dimos.msgs.geometry_msgs.Twist",
    Twist=lambda **kwargs: SimpleNamespace(**kwargs),
)
_module(
    "dimos.msgs.geometry_msgs.Vector3",
    Vector3=lambda *args: args,
)
_module("dimos.msgs.sensor_msgs.Image", Image=_Image)
_module("dimos.msgs.sensor_msgs.PointCloud2", PointCloud2=_PointCloud2)
_module(
    "dimos.robot.unitree.connection",
    UnitreeWebRTCConnection=_UnitreeWebRTCConnection,
)
_module("dimos.utils.logging_config", setup_logger=_setup_logger)
_module("dimos.utils.path_utils", get_project_root=lambda: Path.cwd())
_module("dimos.web.dimos_interface.api.server", FastAPIServer=_FastAPIServer)
_module("unitree_webrtc_connect", constants=SimpleNamespace(RTC_TOPIC={"SPORT_MOD": "sport"}))
_module("unitree_webrtc_connect.constants", RTC_TOPIC={"SPORT_MOD": "sport"})

import iphone_middleware


def test_realtime_session_config_defaults() -> None:
    session = iphone_middleware._build_realtime_session_config(
        model=iphone_middleware.DEFAULT_REALTIME_MODEL,
        voice="echo",
        reasoning_effort=iphone_middleware.DEFAULT_REALTIME_REASONING_EFFORT,
    )

    assert session["type"] == "realtime"
    assert session["model"] == "gpt-realtime-2"
    assert session["output_modalities"] == ["audio"]
    assert session["audio"]["output"]["voice"] == "echo"
    assert session["reasoning"]["effort"] == "low"
    assert "Speak only the exact Fetch dog line" in session["instructions"]


def test_realtime_client_secret_requires_openai_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    middleware = iphone_middleware.FetchIphoneMiddleware(
        tts_provider="openai",
        enable_realtime=True,
    )

    with patch("iphone_middleware.OpenAI") as openai_cls:
        response = TestClient(middleware.server.app).post("/realtime/client-secret")

    openai_cls.assert_not_called()
    assert response.status_code == 503
    assert "OPENAI_API_KEY" in response.json()["error"]


def test_realtime_client_secret_disabled_by_default(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    middleware = iphone_middleware.FetchIphoneMiddleware()

    with patch("iphone_middleware.OpenAI") as openai_cls:
        response = TestClient(middleware.server.app).post("/realtime/client-secret")

    openai_cls.assert_not_called()
    assert response.status_code == 404
    assert "Realtime is disabled" in response.json()["error"]


def test_realtime_client_secret_uses_session_config(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "openai-key")
    middleware = iphone_middleware.FetchIphoneMiddleware(
        tts_provider="openai",
        enable_realtime=True,
        realtime_model="gpt-realtime-2",
        realtime_reasoning_effort="low",
        tts_voice="echo",
    )

    with patch("iphone_middleware.OpenAI") as openai_cls:
        openai_client = openai_cls.return_value
        openai_client.realtime.client_secrets.create.return_value = SimpleNamespace(
            model_dump=lambda mode="json": {"client_secret": {"value": "rt-secret"}}
        )

        response = TestClient(middleware.server.app).post("/realtime/client-secret")

    assert response.status_code == 200
    openai_cls.assert_called_once_with(
        timeout=policy.DEFAULT_REQUEST_TIMEOUT_S,
        max_retries=policy.DEFAULT_MAX_RETRIES,
    )
    call_kwargs = openai_client.realtime.client_secrets.create.call_args.kwargs
    session = call_kwargs["session"]
    assert session["model"] == "gpt-realtime-2"
    assert session["audio"]["output"]["voice"] == "echo"
    assert session["reasoning"]["effort"] == "low"
    assert response.json()["client_secret"]["value"] == "rt-secret"
    assert response.json()["model"] == "gpt-realtime-2"


def test_hello_defaults_to_gemini_speak_route() -> None:
    middleware = iphone_middleware.FetchIphoneMiddleware()

    with TestClient(middleware.server.app).websocket_connect("/fetch/ws") as ws:
        hello = ws.receive_json()

    assert hello["tts_provider"] == "gemini"
    assert hello["audio_route"] == "speak"
    assert hello["realtime_enabled"] is False


def test_cli_defaults_to_gemini_tts(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["iphone_middleware"])

    args = iphone_middleware._parse_args()

    assert args.tts_provider == "gemini"


def test_hello_advertises_gemini_speak_route() -> None:
    middleware = iphone_middleware.FetchIphoneMiddleware(
        tts_provider="gemini",
        enable_realtime=True,
    )

    with TestClient(middleware.server.app).websocket_connect("/fetch/ws") as ws:
        hello = ws.receive_json()

    assert hello["tts_provider"] == "gemini"
    assert hello["audio_route"] == "speak"
    assert hello["realtime_enabled"] is False


def test_hello_advertises_openai_realtime_when_enabled() -> None:
    middleware = iphone_middleware.FetchIphoneMiddleware(
        tts_provider="openai",
        enable_realtime=True,
    )

    with TestClient(middleware.server.app).websocket_connect("/fetch/ws") as ws:
        hello = ws.receive_json()

    assert hello["tts_provider"] == "openai"
    assert hello["audio_route"] == "realtime_then_speak"
    assert hello["realtime_enabled"] is True


def test_speak_uses_gemini_tts_without_openai_client(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    middleware = iphone_middleware.FetchIphoneMiddleware(
        tts_provider="gemini",
        tts_voice="echo",
    )
    gemini_tts = AsyncMock(return_value=b"RIFFwav")

    with (
        patch("iphone_middleware.gemini_live_tts", new=gemini_tts),
        patch("iphone_middleware.OpenAI") as openai_cls,
    ):
        response = TestClient(middleware.server.app).post(
            "/speak",
            json={"text": "Hello beach!"},
        )

    openai_cls.assert_not_called()
    gemini_tts.assert_awaited_once_with(
        "Hello beach!",
        voice="Charon",
    )
    assert response.status_code == 200
    assert response.content == b"RIFFwav"
    assert response.headers["content-type"] == "audio/wav"
