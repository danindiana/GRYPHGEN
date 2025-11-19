"""Kafka topic definitions."""

from enum import Enum


class Topics(str, Enum):
    """Kafka topic names."""

    # Service topics
    CODE_GENERATION = "code-generation"
    CODE_GENERATION_RESPONSE = "code-generation-response"

    AUTOMATED_TESTING = "automated-testing"
    AUTOMATED_TESTING_RESPONSE = "automated-testing-response"

    PROJECT_MANAGEMENT = "project-management"
    PROJECT_MANAGEMENT_RESPONSE = "project-management-response"

    DOCUMENTATION = "documentation"
    DOCUMENTATION_RESPONSE = "documentation-response"

    COLLABORATION = "collaboration"
    COLLABORATION_RESPONSE = "collaboration-response"

    SELF_IMPROVEMENT = "self-improvement"
    SELF_IMPROVEMENT_RESPONSE = "self-improvement-response"

    # System topics
    SYSTEM_METRICS = "system-metrics"
    SYSTEM_LOGS = "system-logs"
    SYSTEM_ALERTS = "system-alerts"

    # User events
    USER_EVENTS = "user-events"
    FEEDBACK_EVENTS = "feedback-events"

    @classmethod
    def get_response_topic(cls, request_topic: str) -> str:
        """Get response topic for a request topic."""
        response_map = {
            cls.CODE_GENERATION: cls.CODE_GENERATION_RESPONSE,
            cls.AUTOMATED_TESTING: cls.AUTOMATED_TESTING_RESPONSE,
            cls.PROJECT_MANAGEMENT: cls.PROJECT_MANAGEMENT_RESPONSE,
            cls.DOCUMENTATION: cls.DOCUMENTATION_RESPONSE,
            cls.COLLABORATION: cls.COLLABORATION_RESPONSE,
            cls.SELF_IMPROVEMENT: cls.SELF_IMPROVEMENT_RESPONSE,
        }
        return response_map.get(request_topic, "")
