"""WebSocket router for real-time features."""

import logging
from typing import Dict, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, Query

from .manager import get_connection_manager
from ..auth.jwt_handler import verify_token

logger = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    token: str = Query(...),
):
    """
    WebSocket endpoint for real-time communication.

    Requires authentication via query parameter token.

    Args:
        websocket: WebSocket connection
        token: JWT authentication token

    Messages format:
        {
            "type": "message_type",
            "payload": {...},
            "room": "optional_room_name"
        }

    Message types:
        - join_room: Join a room
        - leave_room: Leave a room
        - send_message: Send message to room
        - broadcast: Broadcast to all users
    """
    manager = get_connection_manager()

    # Authenticate user
    try:
        payload = verify_token(token)
        user_id = payload.get("sub")

        if not user_id:
            await websocket.close(code=1008, reason="Invalid token")
            return

    except Exception as e:
        logger.error(f"WebSocket authentication failed: {e}")
        await websocket.close(code=1008, reason="Authentication failed")
        return

    # Connect user
    await manager.connect(websocket, user_id)

    try:
        while True:
            # Receive message
            data = await websocket.receive_json()

            # Handle message based on type
            message_type = data.get("type")
            payload = data.get("payload", {})
            room = data.get("room")

            if message_type == "join_room" and room:
                manager.join_room(room, user_id)
                await manager.send_personal_message(
                    {"type": "joined_room", "room": room},
                    user_id,
                )

            elif message_type == "leave_room" and room:
                manager.leave_room(room, user_id)
                await manager.send_personal_message(
                    {"type": "left_room", "room": room},
                    user_id,
                )

            elif message_type == "send_message" and room:
                # Send to room
                message = {
                    "type": "room_message",
                    "room": room,
                    "from": user_id,
                    "payload": payload,
                }
                await manager.send_to_room(room, message)

            elif message_type == "broadcast":
                # Broadcast to all
                message = {
                    "type": "broadcast",
                    "from": user_id,
                    "payload": payload,
                }
                await manager.broadcast(message)

            elif message_type == "ping":
                # Heartbeat
                await manager.send_personal_message(
                    {"type": "pong"},
                    user_id,
                )

    except WebSocketDisconnect:
        manager.disconnect(user_id)
        logger.info(f"User {user_id} disconnected")

    except Exception as e:
        logger.error(f"WebSocket error for user {user_id}: {e}")
        manager.disconnect(user_id)


@router.get("/ws/stats")
async def get_websocket_stats() -> Dict[str, Any]:
    """
    Get WebSocket connection statistics.

    Returns:
        Connection statistics
    """
    manager = get_connection_manager()
    return manager.get_stats()
