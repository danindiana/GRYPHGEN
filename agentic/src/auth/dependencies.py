"""Authentication dependencies for FastAPI."""

from typing import Optional, List
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from .jwt_handler import get_current_user
from ..database.models import User, UserRole

security = HTTPBearer()


async def require_auth(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> User:
    """
    Require authentication.

    Args:
        credentials: Bearer token credentials

    Returns:
        Current user

    Raises:
        HTTPException: If authentication fails
    """
    token = credentials.credentials
    user = await get_current_user(token)
    return user


def require_role(allowed_roles: List[UserRole]):
    """
    Require specific user role.

    Args:
        allowed_roles: List of allowed roles

    Returns:
        Dependency function

    Example:
        ```python
        @app.get("/admin")
        async def admin_endpoint(
            user: User = Depends(require_role([UserRole.ADMIN]))
        ):
            return {"message": "Admin access granted"}
        ```
    """

    async def role_checker(user: User = Depends(require_auth)) -> User:
        if user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Insufficient permissions",
            )
        return user

    return role_checker


# Convenience dependencies for common roles
require_admin = require_role([UserRole.ADMIN])
require_user = require_role([UserRole.USER, UserRole.ADMIN])
