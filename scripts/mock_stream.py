"""Mock Transcript Producer: replay demo Validation calls over the WebSocket (plan §19).

Run:  uv run python -m scripts.mock_stream [--url ws://127.0.0.1:8000/ws] [--delay 0.3] [--call vishing_341]
Sends one segment at a time and prints the running score. Requires the server to be up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import websockets

from src.data.load_korccvi import load_call_manifest, load_transcript, segment_texts

DEMO_CALLS_PATH = Path("experiments/demo_calls.json")


def demo_segments(call_id: str) -> tuple[int, list[str]]:
    record = next(record for record in load_call_manifest("val") if record["call_id"] == call_id)
    texts = [text.strip() for text in segment_texts(load_transcript(record["transcript_filepath"])) if text.strip()]
    return record["label"], texts


async def replay(url: str, call_id: str, delay: float) -> None:
    label, segments = demo_segments(call_id)
    print(f"\n=== {call_id} (정답: {'피싱' if label else '정상'}, {len(segments)} segments) ===")
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"type": "start", "session_id": call_id}))
        for seq, text in enumerate(segments, 1):
            await ws.send(json.dumps({"type": "segment", "session_id": call_id, "seq": seq, "text": text}, ensure_ascii=False))
            reply = json.loads(await ws.recv())
            if reply["type"] == "score":
                bar = "#" * int(reply["model_score"] * 20)
                print(f"seq {seq:3d} | {reply['model_score']:.3f} {bar:20s}| {reply['label']:8s} | {text[:40]}")
            else:
                print(f"seq {seq:3d} | error: {reply['message']}")
            await asyncio.sleep(delay)
        await ws.send(json.dumps({"type": "end", "session_id": call_id}))


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="ws://127.0.0.1:8000/ws")
    parser.add_argument("--delay", type=float, default=0.3)
    parser.add_argument("--call", default=None, help="one call_id; default = all demo calls")
    args = parser.parse_args()
    calls = [args.call] if args.call else [call["call_id"] for call in json.loads(DEMO_CALLS_PATH.read_text(encoding="utf-8"))["calls"]]
    for call_id in calls:
        await replay(args.url, call_id, args.delay)


if __name__ == "__main__":
    asyncio.run(main())
