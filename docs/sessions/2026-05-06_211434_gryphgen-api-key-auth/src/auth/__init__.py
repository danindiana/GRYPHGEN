"""Authentication and authorization for GRYPHGEN Agentic."""

# Lazy imports only — jwt_handler and dependencies pull in database/models
# which has SQLAlchemy issues. Import directly when needed:
#   from .api_keys import require_api_key   (no DB dep)
#   from .jwt_handler import verify_token   (needs DB)

from .api_keys import require_api_key

__all__ = ["require_api_key"]
