"""Kafka consumer implementation."""

import json
import logging
import asyncio
from typing import Dict, Any, Callable, Optional, List, Awaitable
from abc import ABC, abstractmethod

from aiokafka import AIOKafkaConsumer
from aiokafka.errors import KafkaError

from ..common.config import get_settings

logger = logging.getLogger(__name__)


class MessageHandler(ABC):
    """Abstract base class for message handlers."""

    @abstractmethod
    async def handle(self, message: Dict[str, Any]) -> None:
        """
        Handle a message.

        Args:
            message: Deserialized message payload
        """
        pass


class KafkaConsumerManager:
    """
    Asynchronous Kafka consumer manager.

    Manages multiple consumers and message handlers.
    """

    def __init__(
        self,
        group_id: str,
        topics: List[str],
        handler: MessageHandler,
    ):
        """
        Initialize Kafka consumer.

        Args:
            group_id: Consumer group ID
            topics: List of topics to subscribe to
            handler: Message handler instance
        """
        self.settings = get_settings()
        self.group_id = group_id
        self.topics = topics
        self.handler = handler
        self.consumer: Optional[AIOKafkaConsumer] = None
        self._running = False
        self._task: Optional[asyncio.Task] = None

    async def start(self) -> None:
        """Start the Kafka consumer."""
        if self._running:
            return

        try:
            self.consumer = AIOKafkaConsumer(
                *self.topics,
                bootstrap_servers=self.settings.kafka_bootstrap_servers,
                group_id=self.group_id,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='earliest',
                enable_auto_commit=True,
                auto_commit_interval_ms=1000,
            )

            await self.consumer.start()
            self._running = True
            logger.info(f"Kafka consumer started: group={self.group_id}, topics={self.topics}")

            # Start consumption task
            self._task = asyncio.create_task(self._consume())

        except Exception as e:
            logger.error(f"Failed to start Kafka consumer: {e}")
            raise

    async def stop(self) -> None:
        """Stop the Kafka consumer."""
        if not self._running:
            return

        self._running = False

        # Cancel consumption task
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        # Stop consumer
        if self.consumer:
            await self.consumer.stop()

        logger.info("Kafka consumer stopped")

    async def _consume(self) -> None:
        """Consume messages from Kafka."""
        try:
            async for message in self.consumer:
                if not self._running:
                    break

                try:
                    # Process message
                    await self._process_message(message.value)

                except Exception as e:
                    logger.error(f"Error processing message: {e}", exc_info=True)
                    # Continue processing other messages

        except asyncio.CancelledError:
            logger.info("Consumer task cancelled")
            raise
        except Exception as e:
            logger.error(f"Error in consumer loop: {e}", exc_info=True)

    async def _process_message(self, message: Dict[str, Any]) -> None:
        """
        Process a single message.

        Args:
            message: Deserialized message
        """
        message_type = message.get("type", "unknown")

        logger.debug(f"Processing {message_type} message")

        try:
            # Call handler
            await self.handler.handle(message)

        except Exception as e:
            logger.error(f"Handler error: {e}", exc_info=True)
            raise

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


class CodeGenerationHandler(MessageHandler):
    """Handler for code generation messages."""

    async def handle(self, message: Dict[str, Any]) -> None:
        """Handle code generation message."""
        from ..models.code_generator import CodeGeneratorModel
        from ..kafka.producer import get_producer

        if message.get("type") != "request":
            return

        request_id = message.get("request_id")
        response_topic = message.get("response_topic")
        payload = message.get("payload", {})

        try:
            # Generate code
            model = CodeGeneratorModel()
            code = await model.generate(
                prompt=payload.get("prompt"),
                language=payload.get("language", "python"),
            )

            # Send response
            producer = get_producer()
            await producer.send_response(
                response_topic=response_topic,
                request_id=request_id,
                payload={"code": code},
                success=True,
            )

        except Exception as e:
            logger.error(f"Code generation failed: {e}")
            # Send error response
            producer = get_producer()
            await producer.send_response(
                response_topic=response_topic,
                request_id=request_id,
                payload={},
                success=False,
                error=str(e),
            )


class TestGenerationHandler(MessageHandler):
    """Handler for test generation messages."""

    async def handle(self, message: Dict[str, Any]) -> None:
        """Handle test generation message."""
        from ..models.test_generator import TestGeneratorModel
        from ..kafka.producer import get_producer

        if message.get("type") != "request":
            return

        request_id = message.get("request_id")
        response_topic = message.get("response_topic")
        payload = message.get("payload", {})

        try:
            # Generate tests
            model = TestGeneratorModel()
            result = await model.generate_tests(
                source_code=payload.get("source_code"),
                language=payload.get("language", "python"),
                framework=payload.get("framework", "pytest"),
            )

            # Send response
            producer = get_producer()
            await producer.send_response(
                response_topic=response_topic,
                request_id=request_id,
                payload=result,
                success=True,
            )

        except Exception as e:
            logger.error(f"Test generation failed: {e}")
            producer = get_producer()
            await producer.send_response(
                response_topic=response_topic,
                request_id=request_id,
                payload={},
                success=False,
                error=str(e),
            )


class DocumentationHandler(MessageHandler):
    """Handler for documentation generation messages."""

    async def handle(self, message: Dict[str, Any]) -> None:
        """Handle documentation generation message."""
        from ..models.doc_generator import DocumentationGeneratorModel
        from ..kafka.producer import get_producer

        if message.get("type") != "request":
            return

        request_id = message.get("request_id")
        response_topic = message.get("response_topic")
        payload = message.get("payload", {})

        try:
            # Generate documentation
            model = DocumentationGeneratorModel()
            result = await model.generate_documentation(
                source_code=payload.get("source_code"),
                language=payload.get("language", "python"),
                doc_format=payload.get("format", "markdown"),
            )

            # Send response
            producer = get_producer()
            await producer.send_response(
                response_topic=response_topic,
                request_id=request_id,
                payload=result,
                success=True,
            )

        except Exception as e:
            logger.error(f"Documentation generation failed: {e}")
            producer = get_producer()
            await producer.send_response(
                response_topic=response_topic,
                request_id=request_id,
                payload={},
                success=False,
                error=str(e),
            )
