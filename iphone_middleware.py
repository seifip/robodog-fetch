#!/usr/bin/env python3
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

import argparse
import asyncio
import base64
from datetime import datetime
import os
from pathlib import Path
import time
from typing import Any, cast

from dotenv import load_dotenv
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from unitree_webrtc_connect.constants import RTC_TOPIC

from dimos.experimental.fetch.policy import (
    DEFAULT_GEMINI_VISION_MODEL,
    DEFAULT_MAX_RETRIES,
    DEFAULT_OPENAI_VISION_MODEL,
    DEFAULT_REQUEST_TIMEOUT_S,
    FetchPolicy,
    FetchPolicyConfig,
    VisionProvider,
)
from dimos.experimental.fetch.record3d_source import Record3DSource
from dimos.msgs.geometry_msgs.Twist import Twist
from dimos.msgs.geometry_msgs.Vector3 import Vector3
from dimos.robot.unitree.connection import UnitreeWebRTCConnection
from dimos.utils.logging_config import setup_logger
from dimos.utils.path_utils import get_project_root
from dimos.web.dimos_interface.api.server import FastAPIServer

logger = setup_logger()

STATIC_DIR = Path(__file__).parent / "static"
CAPTURE_DIR = STATIC_DIR / "captures"
DEFAULT_PORT = 8455


class FetchIphoneMiddleware:
    """HTTPS phone-camera middleware for testing the Fetch behavior."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = DEFAULT_PORT,
        model: str | None = None,
        vision_provider: VisionProvider = "openai",
        tts_model: str = "tts-1",
        tts_voice: str = "echo",
        record3d: bool = False,
        record3d_device_index: int = 0,
        robot_ip: str | None = None,
        robot_connection_method: str = "local_ap",
    ) -> None:
        self.host = host
        self.port = port
        self.tts_model = tts_model
        self.tts_voice = tts_voice
        self.robot_ip = robot_ip
        self.robot_connection_method = robot_connection_method
        self.policy = FetchPolicy(
            FetchPolicyConfig(model=model, vision_provider=vision_provider)
        )
        self.server = FastAPIServer(
            dev_name="Fetch iPhone Middleware",
            edge_type="Bidirectional",
            host=host,
            port=port,
        )
        self._openai_client: OpenAI | None = None
        self._record3d_source = Record3DSource(record3d_device_index) if record3d else None
        self._setup_routes()

    def _with_robot(self, callback: Any) -> Any:
        if not self.robot_ip:
            return {"enabled": False, "ok": False, "message": "Robot IP is not configured"}
        conn = UnitreeWebRTCConnection(self.robot_ip, connection_method=self.robot_connection_method)
        try:
            return callback(conn)
        finally:
            conn.stop()

    @staticmethod
    def _sport(conn: UnitreeWebRTCConnection, api_id: int, parameter: dict[str, Any] | None = None) -> Any:
        payload: dict[str, Any] = {"api_id": api_id}
        if parameter is not None:
            payload["parameter"] = parameter
        return conn.publish_request(RTC_TOPIC["SPORT_MOD"], payload)

    def _get_openai_client(self) -> OpenAI | None:
        if not os.getenv("OPENAI_API_KEY"):
            return None
        if self._openai_client is None:
            self._openai_client = OpenAI(
                timeout=DEFAULT_REQUEST_TIMEOUT_S,
                max_retries=DEFAULT_MAX_RETRIES,
            )
        return self._openai_client

    def _setup_routes(self) -> None:
        self.server.app.router.routes = [
            route for route in self.server.app.router.routes if getattr(route, "path", None) != "/"
        ]

        @self.server.app.get("/", response_class=HTMLResponse)
        @self.server.app.get("/fetch", response_class=HTMLResponse)
        async def index() -> HTMLResponse:
            return HTMLResponse(
                content=(STATIC_DIR / "index.html").read_text(),
                headers={"Cache-Control": "no-store"},
            )

        @self.server.app.get("/health")
        async def health() -> dict[str, Any]:
            return {
                "ok": True,
                "service": "fetch-iphone",
                "port": self.port,
                "robot_enabled": bool(self.robot_ip),
            }

        @self.server.app.post("/robot/preflight")
        async def robot_preflight() -> Any:
            def run(conn: UnitreeWebRTCConnection) -> dict[str, Any]:
                responses = {
                    "recovery_stand": self._sport(conn, 1006),
                }
                time.sleep(2.0)
                responses["balance_stand"] = self._sport(conn, 1002)
                time.sleep(0.6)
                responses["switch_joystick"] = self._sport(conn, 1027, {"data": True})
                return {"enabled": True, "ok": True, "responses": responses}

            return await asyncio.to_thread(self._with_robot, run)

        @self.server.app.post("/robot/action")
        async def robot_action(payload: dict[str, Any]) -> Any:
            action = str(payload.get("action") or "").strip()

            def run(conn: UnitreeWebRTCConnection) -> dict[str, Any]:
                if action == "move":
                    linear_x = float(payload.get("linear_x") or 0.0)
                    angular_z = float(payload.get("angular_z") or 0.0)
                    duration_s = max(0.0, min(2.0, float(payload.get("duration_s") or 0.0)))
                    twist = Twist(
                        linear=Vector3(linear_x, 0.0, 0.0),
                        angular=Vector3(0.0, 0.0, angular_z),
                    )
                    return {"enabled": True, "ok": conn.move(twist, duration=duration_s)}
                if action == "wave":
                    return {"enabled": True, "ok": True, "response": self._sport(conn, 1016)}
                if action == "dance":
                    return {"enabled": True, "ok": True, "response": self._sport(conn, 1022)}
                if action == "stand":
                    return {"enabled": True, "ok": True, "response": self._sport(conn, 1002)}
                return {"enabled": True, "ok": False, "message": f"Unknown robot action {action!r}"}

            return await asyncio.to_thread(self._with_robot, run)

        @self.server.app.get("/record3d/status")
        async def record3d_status() -> dict[str, Any]:
            if self._record3d_source is None:
                return {"enabled": False}
            return {"enabled": True, **self._record3d_source.status()}

        @self.server.app.post("/record3d/restart")
        async def record3d_restart() -> dict[str, Any]:
            if self._record3d_source is None:
                return {"enabled": False, "restarted": False}
            self._record3d_source.restart()
            return {"enabled": True, "restarted": True, **self._record3d_source.status()}

        @self.server.app.get("/record3d/latest.jpg")
        async def record3d_latest_jpg() -> Response:
            if self._record3d_source is None:
                return JSONResponse({"error": "Record3D is not enabled"}, status_code=404)
            frame = self._record3d_source.latest()
            if frame is None:
                return JSONResponse({"error": "No Record3D frame received yet"}, status_code=404)
            return Response(content=frame.jpeg_bytes, media_type="image/jpeg")

        @self.server.app.get("/record3d/latest-depth.jpg")
        async def record3d_latest_depth_jpg() -> Response:
            if self._record3d_source is None:
                return JSONResponse({"error": "Record3D is not enabled"}, status_code=404)
            frame = self._record3d_source.latest()
            if frame is None:
                return JSONResponse({"error": "No Record3D frame received yet"}, status_code=404)
            return Response(content=frame.depth_jpeg_bytes, media_type="image/jpeg")

        def stream_record3d_jpegs(frame_attr: str) -> StreamingResponse | JSONResponse:
            if self._record3d_source is None:
                return JSONResponse({"error": "Record3D is not enabled"}, status_code=404)

            async def stream_frames() -> Any:
                last_captured_at = 0.0
                while True:
                    frame = self._record3d_source.latest()
                    if frame is not None and frame.captured_at != last_captured_at:
                        last_captured_at = frame.captured_at
                        jpeg_bytes = getattr(frame, frame_attr)
                        yield (
                            b"--frame\r\n"
                            b"Content-Type: image/jpeg\r\n"
                            b"Cache-Control: no-cache\r\n\r\n"
                            + jpeg_bytes
                            + b"\r\n"
                        )
                    await asyncio.sleep(0.03)

            return StreamingResponse(
                stream_frames(),
                media_type="multipart/x-mixed-replace; boundary=frame",
                headers={"Cache-Control": "no-cache"},
            )

        @self.server.app.get("/record3d/stream.mjpg")
        async def record3d_stream() -> Any:
            return stream_record3d_jpegs("jpeg_bytes")

        @self.server.app.get("/record3d/stream-depth.mjpg")
        async def record3d_depth_stream() -> Any:
            return stream_record3d_jpegs("depth_jpeg_bytes")

        @self.server.app.post("/record3d/analyze")
        async def record3d_analyze(payload: dict[str, Any] | None = None) -> Any:
            if self._record3d_source is None:
                return JSONResponse({"error": "Record3D is not enabled"}, status_code=404)
            frame = self._record3d_source.latest()
            if frame is None:
                return JSONResponse({"error": "No Record3D frame received yet"}, status_code=404)
            interaction_phase = str((payload or {}).get("interaction_phase") or "find_guest")
            return await asyncio.to_thread(
                self.policy.analyze_frame,
                frame.image_data_url,
                frame.depth_hint,
                interaction_phase,
            )

        @self.server.app.post("/photos/save")
        async def save_photo(payload: dict[str, Any]) -> Any:
            image_bytes: bytes | None = None
            image_data_url = str(payload.get("image_data_url") or "")

            if image_data_url:
                try:
                    _, encoded = image_data_url.split(",", 1)
                    image_bytes = base64.b64decode(encoded)
                except (ValueError, base64.binascii.Error):
                    return JSONResponse({"error": "Invalid image_data_url"}, status_code=400)
            elif self._record3d_source is not None:
                frame = self._record3d_source.latest()
                if frame is not None:
                    image_bytes = frame.jpeg_bytes

            if image_bytes is None:
                return JSONResponse({"error": "No image available to save"}, status_code=404)

            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            filename = f"fetch-{datetime.now().strftime('%Y%m%d-%H%M%S-%f')}.jpg"
            path = CAPTURE_DIR / filename
            path.write_bytes(image_bytes)
            return {
                "saved": True,
                "url": f"/fetch/static/captures/{filename}",
                "path": str(path),
            }

        @self.server.app.post("/speak")
        async def speak(payload: dict[str, Any]) -> Response:
            text = str(payload.get("text") or "").strip()
            if not text:
                return JSONResponse({"error": "Missing text"}, status_code=400)
            if len(text) > 240:
                return JSONResponse({"error": "Text is too long"}, status_code=400)

            openai_client = self._get_openai_client()
            if openai_client is None:
                return JSONResponse(
                    {"error": "OPENAI_API_KEY is not set; speech requires OpenAI TTS"},
                    status_code=503,
                )

            speech = openai_client.audio.speech.create(
                model=self.tts_model,
                voice=str(payload.get("voice") or self.tts_voice),
                input=text,
                response_format="mp3",
            )
            return Response(content=speech.content, media_type="audio/mpeg")

        if STATIC_DIR.is_dir():
            self.server.app.mount(
                "/fetch/static",
                StaticFiles(directory=str(STATIC_DIR)),
                name="fetch_static",
            )

        @self.server.app.websocket("/fetch/ws")
        async def websocket_endpoint(ws: WebSocket) -> None:
            await ws.accept()
            await ws.send_json(
                {
                    "type": "hello",
                    "service": "fetch-iphone",
                    "model": self.policy.config.model,
                    "vision_provider": self.policy.config.vision_provider,
                }
            )
            logger.info("Fetch iPhone client connected")
            try:
                while True:
                    message = await ws.receive_json()
                    message_type = message.get("type")
                    if message_type == "record3d_frame":
                        if self._record3d_source is None:
                            await ws.send_json(
                                {"type": "error", "message": "Record3D is not enabled"}
                            )
                            continue
                        frame = self._record3d_source.latest()
                        if frame is None:
                            await ws.send_json(
                                {"type": "error", "message": "No Record3D frame received yet"}
                            )
                            continue
                        decision = await asyncio.to_thread(
                            self.policy.analyze_frame,
                            frame.image_data_url,
                            frame.depth_hint,
                            str(message.get("interaction_phase") or "find_guest"),
                        )
                        decision["frame_id"] = message.get("frame_id")
                        decision["record3d"] = self._record3d_source.status()
                        await ws.send_json(decision)
                        continue

                    if message_type != "frame":
                        await ws.send_json(
                            {"type": "error", "message": "Expected frame or record3d_frame message"}
                        )
                        continue

                    image_data_url = str(message.get("image") or "")
                    depth_hint = message.get("depth_hint")
                    decision = await asyncio.to_thread(
                        self.policy.analyze_frame,
                        image_data_url,
                        depth_hint if isinstance(depth_hint, dict) else None,
                        str(message.get("interaction_phase") or "find_guest"),
                    )
                    decision["frame_id"] = message.get("frame_id")
                    await ws.send_json(decision)
            except WebSocketDisconnect:
                logger.info("Fetch iPhone client disconnected")
            except Exception as exc:
                logger.exception("Fetch WebSocket error")
                try:
                    await ws.send_json({"type": "error", "message": str(exc)})
                except Exception:
                    pass

    def run(self, ssl: bool = True) -> None:
        if self._record3d_source is not None:
            self._record3d_source.start()
        if ssl:
            certs_dir = get_project_root() / "assets" / "teleop_certs"
            self.server.run(ssl=True, ssl_certs_dir=certs_dir)
        else:
            self.server.run()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Fetch iPhone middleware.")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host for the HTTPS server.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Bind port.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Vision model. Defaults to "
            f"{DEFAULT_OPENAI_VISION_MODEL} for OpenAI and "
            f"{DEFAULT_GEMINI_VISION_MODEL} for Gemini."
        ),
    )
    parser.add_argument(
        "--vision-provider",
        choices=("openai", "gemini"),
        default="openai",
        help="Provider for image analysis.",
    )
    parser.add_argument("--tts-model", default="tts-1", help="OpenAI TTS model.")
    parser.add_argument("--tts-voice", default="echo", help="OpenAI TTS voice.")
    parser.add_argument("--record3d", action="store_true", help="Read RGBD frames from Record3D over USB.")
    parser.add_argument("--record3d-device-index", type=int, default=0, help="Record3D device index.")
    parser.add_argument("--robot-ip", default=None, help="Optional live Unitree Go2 IP for Fetch actions.")
    parser.add_argument(
        "--robot-connection-method",
        default="local_ap",
        help="Unitree WebRTC connection method: auto, local_ap, or local_sta.",
    )
    parser.add_argument("--no-ssl", action="store_true", help="Disable HTTPS for local debugging.")
    args = parser.parse_args()
    try:
        resolved_config = FetchPolicyConfig(
            model=args.model,
            vision_provider=cast(VisionProvider, args.vision_provider),
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.model = resolved_config.model
    args.vision_provider = resolved_config.vision_provider
    return args


def main() -> None:
    load_dotenv(get_project_root() / ".env")
    load_dotenv()
    args = _parse_args()
    middleware = FetchIphoneMiddleware(
        host=args.host,
        port=args.port,
        model=args.model,
        vision_provider=cast(VisionProvider, args.vision_provider),
        tts_model=args.tts_model,
        tts_voice=args.tts_voice,
        record3d=args.record3d,
        record3d_device_index=args.record3d_device_index,
        robot_ip=args.robot_ip,
        robot_connection_method=args.robot_connection_method,
    )
    scheme = "http" if args.no_ssl else "https"
    logger.info(f"Fetch iPhone middleware running at {scheme}://{args.host}:{args.port}/fetch")
    middleware.run(ssl=not args.no_ssl)


if __name__ == "__main__":
    main()
