# Fetch Prototype

This prototype turns the dog behavior into a testable state machine before the
robot is available. It reuses the same DimOS web pattern as phone teleop:
a FastAPI server exposes an HTTPS phone UI, the phone streams camera frames over
WebSocket, and the server returns motion/speech/photo decisions.

## Behavior

1. Run the robot preflight: recovery stand, balance stand, and joystick handoff.
2. Search for a happy or relaxed beachgoer who is lying down or reclining.
3. Reject people who appear busy, holding a phone, holding a drink, reading, working, or using a laptop/tablet.
4. Approach only if the path looks safe.
5. Stop once the target is within 4 meters.
6. Wave and say a personalized joke based on visible, non-sensitive appearance/context.
7. Offer a Coke/drink in exchange for an instant photo.
8. Wait until the person is holding the bottle and both the person and bottle are well framed.
9. Say a cheese cue, save the photo, and trigger a dance.

The prototype does not infer identity or sensitive traits. Humor is constrained to
visible beach context such as hats, sunglasses, towels, heat, shade, and posture.

## Run

```bash
cd /Users/seifip/GitHub/fetch/dimos
python -m dimos.experimental.fetch.iphone_middleware --host 0.0.0.0 --port 8455
```

For Record3D USB RGBD input:

```bash
cd /Users/seifip/GitHub/fetch/dimos
python -m dimos.experimental.fetch.iphone_middleware --host 0.0.0.0 --port 8455 --record3d
```

For a live Go2 on the dog Wi-Fi:

```bash
cd /Users/seifip/GitHub/fetch/dimos
python -m dimos.experimental.fetch.iphone_middleware \
  --host 0.0.0.0 \
  --port 8455 \
  --record3d \
  --robot-ip 192.168.12.1 \
  --robot-connection-method local_ap
```

Open this on the iPhone or Mac:

```text
http://127.0.0.1:8455/fetch
```

Tap `Record` to start the 10 second scan loop.

## iPhone Depth

The browser version streams RGB camera frames and can play OpenAI-generated audio.
Safari does not expose raw LiDAR depth to JavaScript, so true iPhone LiDAR testing
uses the optional Record3D USB adapter. Start Record3D with USB streaming enabled,
press the red record toggle, then run the middleware with `--record3d`.

The adapter exposes:

- `GET /record3d/status`
- `GET /record3d/latest.jpg`
- `GET /record3d/latest-depth.jpg`
- `GET /record3d/stream.mjpg`
- `GET /record3d/stream-depth.mjpg`
- `POST /record3d/restart`
- `POST /record3d/analyze`
- `POST /photos/save`
- `POST /robot/preflight`
- `POST /robot/action`

Record3D frames are analyzed with the RGB image plus a `depth_hint` containing
center-depth and near-object summaries in meters. The default OpenAI vision model
is `gpt-5-mini`.

## Gemini Vision

The fetch policy can use Gemini for image analysis through Gemini's
OpenAI-compatible API, without LangChain:

```bash
export GEMINI_API_KEY=<YOUR_KEY>
python -m dimos.experimental.fetch.iphone_middleware \
  --host 0.0.0.0 \
  --port 8455 \
  --vision-provider gemini
```

Gemini defaults to `gemini-3.5-flash`; pass `--model` only to override it.
`GOOGLE_API_KEY` is also accepted as a fallback for Gemini. The `/speak` endpoint
still uses OpenAI TTS, so set `OPENAI_API_KEY` only if you want browser audio.

## WebSocket Frame

```json
{
  "type": "frame",
  "frame_id": 1,
  "image": "data:image/jpeg;base64,...",
  "depth_hint": null,
  "ts": 1779897600000
}
```

## Decision

```json
{
  "type": "decision",
  "state": "approach",
  "candidate_found": true,
  "target": {
    "bearing": "center",
    "range": "near",
    "description": "person reclining on a towel",
    "free_hand_evidence": "both hands are visible and empty",
    "busy_signals": ["none"]
  },
  "simulated_cmd_vel": {
    "linear_x": 0.22,
    "angular_z": 0.0,
    "duration_s": 0.9
  },
  "action": "wave_offer",
  "photo_ready": false,
  "bottle_visible": false,
  "framing": {
    "person_visible": true,
    "bottle_visible": false,
    "well_framed": false,
    "notes": ""
  },
  "line": "Your pink umbrella is doing heroic shade work. Want a Coke for one instant photo?"
}
```

## Robot Adapter Later

The `simulated_cmd_vel` output maps directly to the Go2 velocity path used in the
direct WebRTC scripts:

- `linear_x`: forward velocity.
- `angular_z`: yaw velocity.
- `duration_s`: command duration.

On the dog, use LiDAR/depth for the final `<4m` stop condition and keep obstacle
avoidance enabled.
