from fastapi import HTTPException, Depends
from fastapi_decorators import depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from SharedBackend.managers import BaseKeyManager


class SDKMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, key_manager: "BaseKeyManager"):
        super().__init__(app)
        self.key_manager = key_manager

    async def dispatch(self, request: Request, call_next):
        api_key = request.headers.get("X-API-Key", "")
        scopes = await self.key_manager.verify_key(api_key)
        request.state.scopes = scopes
        response = await call_next(request)
        return response

    @staticmethod
    def authorize(*allowed_scopes: str):
        allowed_scopes = set(allowed_scopes)

        async def wrapper(request: Request):
            scopes = request.state.scopes
            if scopes is None or not allowed_scopes.intersection(scopes):
                raise HTTPException(status_code=403, detail="Unauthorized")

        return depends(Depends(wrapper))
