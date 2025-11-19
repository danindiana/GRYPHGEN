"""WebSocket support for real-time collaboration."""

from .manager import ConnectionManager, get_connection_manager
from .router import router as websocket_router

__all__ = [
    "ConnectionManager",
    "get_connection_manager",
    "websocket_router",
]
