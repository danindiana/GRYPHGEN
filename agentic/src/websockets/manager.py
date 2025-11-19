"""WebSocket connection manager."""

import logging
import json
from typing import Dict, Set, Any
from functools import lru_cache

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections for real-time features.

    Supports:
    - Broadcasting messages to all connections
    - Sending messages to specific users
    - Room-based messaging
    """

    def __init__(self):
        """Initialize connection manager."""
        # Active connections by user_id
        self.active_connections: Dict[str, WebSocket] = {}

        # Rooms: room_name -> set of user_ids
        self.rooms: Dict[str, Set[str]] = {}

        logger.info("ConnectionManager initialized")

    async def connect(self, websocket: WebSocket, user_id: str) -> None:
        """
        Connect a WebSocket.

        Args:
            websocket: WebSocket connection
            user_id: User identifier
        """
        await websocket.accept()
        self.active_connections[user_id] = websocket
        logger.info(f"User {user_id} connected. Total connections: {len(self.active_connections)}")

    def disconnect(self, user_id: str) -> None:
        """
        Disconnect a WebSocket.

        Args:
            user_id: User identifier
        """
        if user_id in self.active_connections:
            del self.active_connections[user_id]

        # Remove from all rooms
        for room_users in self.rooms.values():
            room_users.discard(user_id)

        logger.info(f"User {user_id} disconnected. Total connections: {len(self.active_connections)}")

    async def send_personal_message(self, message: Dict[str, Any], user_id: str) -> None:
        """
        Send message to a specific user.

        Args:
            message: Message payload
            user_id: Target user ID
        """
        if user_id in self.active_connections:
            websocket = self.active_connections[user_id]
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error sending message to {user_id}: {e}")
                self.disconnect(user_id)

    async def broadcast(self, message: Dict[str, Any]) -> None:
        """
        Broadcast message to all connections.

        Args:
            message: Message payload
        """
        disconnected = []

        for user_id, websocket in self.active_connections.items():
            try:
                await websocket.send_json(message)
            except Exception as e:
                logger.error(f"Error broadcasting to {user_id}: {e}")
                disconnected.append(user_id)

        # Clean up disconnected users
        for user_id in disconnected:
            self.disconnect(user_id)

    def join_room(self, room: str, user_id: str) -> None:
        """
        Add user to a room.

        Args:
            room: Room name
            user_id: User ID
        """
        if room not in self.rooms:
            self.rooms[room] = set()

        self.rooms[room].add(user_id)
        logger.info(f"User {user_id} joined room {room}")

    def leave_room(self, room: str, user_id: str) -> None:
        """
        Remove user from a room.

        Args:
            room: Room name
            user_id: User ID
        """
        if room in self.rooms:
            self.rooms[room].discard(user_id)

            # Remove empty rooms
            if not self.rooms[room]:
                del self.rooms[room]

            logger.info(f"User {user_id} left room {room}")

    async def send_to_room(self, room: str, message: Dict[str, Any]) -> None:
        """
        Send message to all users in a room.

        Args:
            room: Room name
            message: Message payload
        """
        if room not in self.rooms:
            return

        for user_id in self.rooms[room]:
            await self.send_personal_message(message, user_id)

    def get_room_users(self, room: str) -> Set[str]:
        """
        Get all users in a room.

        Args:
            room: Room name

        Returns:
            Set of user IDs
        """
        return self.rooms.get(room, set()).copy()

    def get_stats(self) -> Dict[str, Any]:
        """
        Get connection statistics.

        Returns:
            Statistics dictionary
        """
        return {
            "total_connections": len(self.active_connections),
            "total_rooms": len(self.rooms),
            "rooms": {
                room: len(users)
                for room, users in self.rooms.items()
            },
        }


@lru_cache()
def get_connection_manager() -> ConnectionManager:
    """
    Get singleton connection manager.

    Returns:
        ConnectionManager instance
    """
    return ConnectionManager()
