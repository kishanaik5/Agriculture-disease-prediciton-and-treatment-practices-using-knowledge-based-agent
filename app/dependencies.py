from fastapi import Request, HTTPException

from app.config import get_settings

settings = get_settings()


def master_key_dependency(request: "Request"):
    api_key = request.headers.get("X-API-Key", "")
    if api_key != settings.master_api_key:
        raise HTTPException(status_code=403, detail="Forbidden")

def require_auth(request: Request) -> str:
    """Dependency to require authentication on routes."""
    if not request.state.upstreamId:
        raise HTTPException(status_code=401, detail="Authentication required")
    return request.state.upstreamId


def get_current_user_id(request: Request) -> str | None:
    """Dependency to get current user ID (optional auth)."""
    return getattr(request.state, 'upstreamId', None)
