"""Kafka integration for event-driven architecture."""

from .producer import KafkaProducer, get_producer
from .consumer import KafkaConsumerManager, MessageHandler
from .topics import Topics

__all__ = [
    "KafkaProducer",
    "get_producer",
    "KafkaConsumerManager",
    "MessageHandler",
    "Topics",
]
