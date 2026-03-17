"""WebSocket handlers for streaming inference."""
from __future__ import annotations

import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from adapters.inference.ollama_adapter import OllamaAdapter
from core.entities import InferenceRequest

router = APIRouter()


@router.websocket("/ws/inference/stream")
async def inference_stream(websocket: WebSocket):
    """WebSocket endpoint for streaming inference."""
    await websocket.accept()

    adapter = OllamaAdapter()

    try:
        while True:
            # Receive request JSON
            data = await websocket.receive_text()
            request_data = json.loads(data)

            # Create InferenceRequest
            request = InferenceRequest(
                model=request_data.get("model"),
                messages=request_data.get("messages"),
                temperature=request_data.get("temperature", 0.7),
                max_tokens=request_data.get("max_tokens"),
                top_p=request_data.get("top_p", 1.0),
                top_k=request_data.get("top_k", 50),
                stop=request_data.get("stop"),
                system=request_data.get("system"),
                stream=True,
            )

            # Stream tokens back
            async for token in adapter.stream(request):
                await websocket.send_json({"type": "token", "content": token})

            # Send completion signal
            await websocket.send_json({"type": "done"})

    except WebSocketDisconnect:
        print("WebSocket disconnected")
    except Exception as e:
        await websocket.send_json({"type": "error", "error": str(e)})
    finally:
        await adapter.close()
