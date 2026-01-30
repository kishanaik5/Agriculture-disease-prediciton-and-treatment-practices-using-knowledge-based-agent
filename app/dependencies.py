from fastapi import Request, HTTPException

from app.config import get_settings

settings = get_settings()


def master_key_dependency(request: "Request"):
    api_key = request.headers.get("X-API-Key", "")
    if api_key != settings.master_api_key:
        raise HTTPException(status_code=403, detail="Forbidden")

def require_auth(request: Request) -> str:
    """Dependency to require authentication on routes."""
    # Try getting from state (set by middleware)
    upstream_id = getattr(request.state, "upstreamId", None)
    
    # If not in state, try header directly
    if not upstream_id:
        upstream_id = request.headers.get("X-Upstream-Id")
        # Optional: set it back to state for other dependencies
        if upstream_id:
            request.state.upstreamId = upstream_id

    if not upstream_id:
        raise HTTPException(status_code=401, detail="Authentication required")
        
    return upstream_id


def get_current_user_id(request: Request) -> str | None:
    """Dependency to get current user ID (optional auth)."""
    upstream_id = getattr(request.state, 'upstreamId', None)
    if not upstream_id:
        upstream_id = request.headers.get("X-Upstream-Id")
    return upstream_id
