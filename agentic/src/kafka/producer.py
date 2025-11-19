"""Kafka producer implementation."""

import json
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from functools import lru_cache

from aiokafka import AIOKafkaProducer
from aiokafka.errors import KafkaError

from ..common.config import get_settings

logger = logging.getLogger(__name__)


class KafkaProducer:
    """
    Asynchronous Kafka producer for publishing events.

    Uses aiokafka for high-performance async message production.
    """

    def __init__(self):
        """Initialize Kafka producer."""
        self.settings = get_settings()
        self.producer: Optional[AIOKafkaProducer] = None
        self._started = False

    async def start(self) -> None:
        """Start the Kafka producer."""
        if self._started:
            return

        try:
            self.producer = AIOKafkaProducer(
                bootstrap_servers=self.settings.kafka_bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                compression_type='gzip',
                max_batch_size=16384,
                linger_ms=10,  # Wait 10ms to batch messages
            )

            await self.producer.start()
            self._started = True
            logger.info(f"Kafka producer started: {self.settings.kafka_bootstrap_servers}")

        except Exception as e:
            logger.error(f"Failed to start Kafka producer: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Kafka producer."""
        if self.producer and self._started:
            await self.producer.stop()
            self._started = False
            logger.info("Kafka producer stopped")

    async def send_message(
        self,
        topic: str,
        message: Dict[str, Any],
        key: Optional[str] = None,
    ) -> None:
        """
        Send a message to a Kafka topic.

        Args:
            topic: Topic name
            message: Message payload (will be JSON serialized)
            key: Optional message key for partitioning

        Raises:
            KafkaError: If message send fails
        """
        if not self._started:
            await self.start()

        try:
            # Add metadata to message
            enriched_message = {
                **message,
                "_timestamp": datetime.utcnow().isoformat(),
                "_topic": topic,
            }

            # Send message
            key_bytes = key.encode('utf-8') if key else None

            await self.producer.send_and_wait(
                topic=topic,
                value=enriched_message,
                key=key_bytes,
            )

            logger.debug(f"Sent message to topic '{topic}': {message.get('type', 'unknown')}")

        except KafkaError as e:
            logger.error(f"Failed to send message to '{topic}': {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error sending message: {e}")
            raise

    async def send_request(
        self,
        request_topic: str,
        response_topic: str,
        request_id: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Send a request message with response routing.

        Args:
            request_topic: Topic to send request to
            response_topic: Topic for response
            request_id: Unique request identifier
            payload: Request payload
        """
        message = {
            "type": "request",
            "request_id": request_id,
            "response_topic": response_topic,
            "payload": payload,
        }

        await self.send_message(request_topic, message, key=request_id)

    async def send_response(
        self,
        response_topic: str,
        request_id: str,
        payload: Dict[str, Any],
        success: bool = True,
        error: Optional[str] = None,
    ) -> None:
        """
        Send a response message.

        Args:
            response_topic: Topic to send response to
            request_id: Original request identifier
            payload: Response payload
            success: Whether request was successful
            error: Error message if failed
        """
        message = {
            "type": "response",
            "request_id": request_id,
            "success": success,
            "payload": payload,
        }

        if error:
            message["error"] = error

        await self.send_message(response_topic, message, key=request_id)

    async def send_event(
        self,
        topic: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """
        Send an event message.

        Args:
            topic: Topic to send event to
            event_type: Type of event
            payload: Event payload
        """
        message = {
            "type": "event",
            "event_type": event_type,
            "payload": payload,
        }

        await self.send_message(topic, message)

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


@lru_cache()
def get_producer() -> KafkaProducer:
    """
    Get singleton Kafka producer instance.

    Returns:
        KafkaProducer instance
    """
    return KafkaProducer()
