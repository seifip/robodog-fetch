<div align="center">

<img src="static/logo.png" alt="Fetch" width="240" />

# Fetch

**A robot dog that trades ice-cold Cokes for instant photos.**

[Demo video](#) · [Quickstart](#quickstart-run-it-yourself) · [How it works](#how-it-works-at-a-glance) · [Technical reference](#technical-reference)

</div>

> Replace the demo-video link above with the final recording before submitting.

---

## What it is

Fetch is a Unitree Go2 robot dog that works a crowd like a tiny, soda-carrying
street performer. It wanders, finds someone who looks relaxed and up for a moment
of fun, trots over, cracks a joke about what it sees, and offers them a cold Coke
from its back — all it asks in return is to take their photo.

Under the hood, a vision LLM "reads the room" on every camera frame and decides
where to move, what to say, and when to take the shot. It runs as a single
FastAPI + WebSocket server, so you can try the whole behavior from a phone browser
**before any robot is involved**.

## The experience

1. **Wakes up.** The dog runs its preflight (recovery stand → balance stand →
   joystick handoff) and starts looking around.
2. **Reads the room.** It turns in place scanning for a good guest: someone
   centered in frame, facing the dog, who looks chill, curious, playful, or just
   thirsty. People glued to a phone, laptop, or meal read as "busy — skip."
3. **Approaches — carefully.** If the path looks clear, it walks over and stops
   once the person is within ~4 meters.
4. **Breaks the ice.** It waves and delivers a personalized one-liner riffing on
   what's actually visible in the scene — never on who the person is.
5. **Makes the trade.** "Grab a Coke from my back, then pose for an instant photo."
   It coaches them to hold the Coke up and center themselves in frame.
6. **Takes the shot.** When the framing is right, it snaps the photo, plays a
   Polaroid print sound, tells them it's ready — and dances.

Fetch's comedy voice is confessional, observational, self-deprecating, and mildly
exasperated by the absurdity of being a tiny robot dog hauling soda around a beach.

## How it works (at a glance)

```
  camera frame                vision LLM                 decision                 act
 ┌────────────┐   image    ┌────────────┐   JSON    ┌────────────────┐       ┌──────────┐
 │  phone cam │──(+depth)──▶│  OpenAI /  │──────────▶│ state, cmd_vel,│──────▶│ move Go2 │
 │  Record3D  │            │   Gemini   │           │ line, photo?   │       │ speak    │
 │  live Go2  │            └────────────┘           └────────────────┘       │ snap photo│
 └────────────┘                                                              └──────────┘
        ▲                                                                          │
        └──────────────────────── ~1s scan loop ──────────────────────────────────┘
```

Fetch reuses the **DimOS teleop web pattern**: a FastAPI server serves an HTTPS
phone UI, the phone streams camera frames over a WebSocket, and the server returns
motion / speech / photo decisions. Three camera sources plug into the same loop:

- **Phone browser camera** — zero hardware; the fastest way to try the behavior.
- **Record3D USB (RGBD)** — real iPhone LiDAR depth, since Safari won't expose raw
  depth to JavaScript.
- **Live Unitree Go2** — WebRTC camera + LiDAR over the dog's Wi-Fi.

## Quickstart (run it yourself)

> **Prerequisite:** Fetch lives in the DimOS monorepo at `dimos/experimental/fetch/`
> and imports DimOS modules, so run it from the monorepo root (the package must be on
> your `PYTHONPATH`). Set whichever provider keys you'll use in `.env`:
> `OPENAI_API_KEY`, `GEMINI_API_KEY` (or `GOOGLE_API_KEY`), `CARTESIA_API_KEY`.

**1. No hardware — phone or laptop browser camera:**

```bash
# from the DimOS monorepo root
python -m dimos.experimental.fetch.iphone_middleware --host 0.0.0.0 --port 8455
```

Open `http://127.0.0.1:8455/fetch` and tap **Record** to start the ~1-second scan
loop. (`localhost`/`127.0.0.1` are secure contexts, so HTTPS is optional there; a
phone-over-LAN demo needs HTTPS — the default — or the browser blocks the camera/mic.)

**2. Real iPhone LiDAR depth via Record3D USB:**

```bash
# start Record3D with USB streaming enabled and the red record toggle on, then:
python -m dimos.experimental.fetch.iphone_middleware --host 0.0.0.0 --port 8455 --record3d
```

**3. Live Unitree Go2 on the dog's Wi-Fi:**

```bash
python -m dimos.experimental.fetch.iphone_middleware \
  --host 0.0.0.0 --port 8455 \
  --vision-provider gemini \
  --robot-ip 192.168.12.1 --robot-connection-method local_ap
```

Vision defaults to OpenAI `gpt-5-mini`; `--vision-provider gemini` uses
`gemini-3.5-flash` through Gemini's OpenAI-compatible API (no LangChain). Use
`--no-ssl` for quick local debugging.

## Voice & conversation

Fetch can either **talk at** people (one-way TTS) or **talk with** them (two-way).

**One-way TTS** (`/speak`) supports three providers, switchable at runtime from the
phone UI's **Audio** button — no restart needed:

| Provider | Model | Notes |
|---|---|---|
| **Cartesia Sonic** (default) | `sonic-3.5-2026-05-04` | Lowest latency; needs `CARTESIA_API_KEY` |
| **Gemini Live TTS** | `gemini-3.1-flash-live-preview` | Expressive; needs `GEMINI_API_KEY`/`GOOGLE_API_KEY` |
| **OpenAI TTS** | `tts-1` | Needs `OPENAI_API_KEY` |

OpenAI Realtime WebRTC is an opt-in extra (`--tts-provider openai --enable-realtime`)
that the browser tries first and falls back to `/speak` if it fails. The UI also has a
**Direct/Staged** approach toggle — Direct keeps the fast happy-path greet; Staged
inserts a short settle/stand beat before the wave.

**Two-way conversation** (`--conversation-mode gemini_live`) turns the greeting into a
real voice exchange. Once Fetch reaches a person, the browser opens the mic and streams
audio to a persistent Gemini Live session; turn-taking and barge-in use the Live API's
server-side voice activity detection. The model drives the dog through tool calls —
`accept_offer`, `take_photo`, `celebrate`, `do_trick`, `stop_and_reset` — and photo
framing is fed back in as hints so the spoken coaching matches what the camera sees.

```bash
python -m dimos.experimental.fetch.iphone_middleware \
  --host 0.0.0.0 --port 8455 \
  --vision-provider gemini --conversation-mode gemini_live \
  --robot-ip 192.168.12.1 --robot-connection-method local_ap
```

Want to compare provider latency for yourself? `scripts/latency_bench.py` measures
real round-trip times for the vision and TTS models Fetch uses across whichever keys
you have set:

```bash
python3 scripts/latency_bench.py            # 3 runs each
python3 scripts/latency_bench.py --runs 5
```

## Safety & privacy

- **No identity or sensitive-trait inference.** Humor is constrained to *visible,
  non-sensitive context* — setting, posture, lighting, colors, nearby objects, what's
  happening in the scene.
- **Obstacle-aware approach.** It only moves when the path looks safe and uses
  LiDAR/depth for the final `<4m` stop.
- **Local-demo auth caveat.** `/realtime/client-secret` is intentionally
  unauthenticated for local live demos and is disabled by default. Add an access gate
  before exposing this server on a shared or public network.

## Technical reference

<details>
<summary><b>State machine & interaction phases</b></summary>

```
search → approach → greet → wait_for_coke → photo_ready
  ↑                                              │
  └──────────────── skip ◀───────────────────────┘
```

- **search** — scan left/right (`angular_z != 0`) for a candidate.
- **approach** — move toward the target (`linear_x > 0`, bearing-based `angular_z`).
- **greet** (inside 4 m) — wave and deliver the personalized joke/offer.
- **wait_for_coke** — person hasn't framed the Coke yet; coach them.
- **photo_ready** — Coke + person well-framed; take the photo and dance.
- **skip** — unsafe or blocked; resume searching.

Two interaction phases drive different prompts: `find_guest` (locate and evaluate a
person) and `confirm_coke` (check the person is holding the Coke and ready for a photo).

</details>

<details>
<summary><b>File roles</b></summary>

| File | Role |
|---|---|
| `policy.py` | Core state machine. `FetchPolicy.analyze_frame()` sends image + prompt to the vision LLM and normalizes the JSON response into a decision dict. |
| `iphone_middleware.py` | FastAPI server. WebSocket frame routing (browser / Record3D / Go2) + REST endpoints for robot commands, TTS, Realtime secrets, and photo capture. |
| `conversation.py` | `LiveConversationSession` — the persistent two-way Gemini Live voice session with tool calling. |
| `conversation_prompt.py` | Conversation persona, menu, and safety rules (mirrors `policy.py`). |
| `record3d_source.py` | Background thread reading RGBD frames from Record3D USB; produces JPEG + depth hints. |
| `tts.py` | TTS provider helpers: Gemini Live TTS, voice-name mapping, PCM→WAV conversion. |
| `static/index.html` | Single-page phone UI: camera feed, previews, controls, decision display, audio routing, photo flow. |

</details>

<details>
<summary><b>WebSocket frame → decision</b></summary>

The browser/source sends frames:

```json
{ "type": "frame", "frame_id": 1, "image": "data:image/jpeg;base64,...", "depth_hint": null, "ts": 1779897600000 }
```

The server replies with a decision:

```json
{
  "type": "decision",
  "state": "approach",
  "candidate_found": true,
  "target": { "bearing": "center", "range": "near", "description": "person smiling toward the dog" },
  "simulated_cmd_vel": { "linear_x": 0.22, "angular_z": 0.0, "duration_s": 0.9 },
  "action": "wave_offer",
  "photo_ready": false,
  "framing": { "person_visible": true, "coke_visible": false, "well_framed": false, "notes": "" },
  "line": "That counter stance says you're ready for the tiny robot VIP treatment. Grab a Coke from my back first."
}
```

</details>

<details>
<summary><b>REST endpoints & robot mapping</b></summary>

```
GET  /record3d/status            POST /record3d/restart      POST /robot/preflight
GET  /record3d/latest.jpg        POST /record3d/analyze      POST /robot/action
GET  /record3d/latest-depth.jpg  POST /photos/save
GET  /record3d/stream.mjpg       POST /realtime/client-secret
GET  /record3d/stream-depth.mjpg
```

`simulated_cmd_vel` maps directly onto the Go2 velocity path (despite the name, it's
not simulation-only): `linear_x` = forward velocity, `angular_z` = yaw, `duration_s` =
command duration. On the dog, LiDAR/depth enforces the final `<4m` stop and obstacle
avoidance stays enabled. Saved photos are written to `static/captures/` (gitignored);
set `FETCH_PHOTO_MIRROR_DIRS` (an `os.pathsep`-separated list of folders) to also copy
each photo into e.g. an iCloud Drive or Google Drive folder so the demo phone syncs it.

</details>

## Built on DimOS

Fetch is a DimOS package: it lives at `dimos/experimental/fetch/` in the
[DimOS](https://github.com/dimensionalOS/dimos) monorepo and reuses DimOS primitives
(Unitree WebRTC control, the teleop web/cert pattern, LiDAR). This standalone
`robodog-fetch` repo mirrors those files so the behavior can be developed and tested
on its own; the `python -m dimos.experimental.fetch.iphone_middleware` commands run
from the monorepo root.

## Tests

```bash
pytest -q            # all tests; providers are mocked — no real API calls
pytest test_policy.py
```
