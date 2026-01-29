from collections import defaultdict

from fastapi import Depends, HTTPException
from fastapi_decorators import depends
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request

from SharedBackend.managers import EntityManager


class EntityMiddleware(BaseHTTPMiddleware):
    SELF = "me"

    def __init__(self, app, entity_manager: "EntityManager"):
        super().__init__(app)
        self.entity_manager = entity_manager

    async def dispatch(self, request: Request, call_next):
        if not hasattr(request.state, "upstreamId"): request.state.upstreamId = request.headers.get("X-UPID")
        else: request.state.upstreamId = None
        links = await self.entity_manager.fetch_links(request.state.upstreamId)  # todo: do lazy loading
        request.state.links = links
        response = await call_next(request)
        return response

    @classmethod
    def authorize(cls, *owned_params: tuple[str, str]):
        _owned_params_dict = defaultdict(list)
        for _param, _tablename in owned_params: _owned_params_dict[_param].append(_tablename)
        owned_params = _owned_params_dict

        async def wrapper(request: Request):
            links = request.state.links
            params = dict(request.path_params)
            for param, tablenames in owned_params.items():
                if param not in params: continue
                link = []
                for tablename in tablenames:
                    if tablename in links:
                        link = links[tablename]
                        break
                if params[param] == cls.SELF:
                    pos = int(request.query_params.get(f"pos{param[0].upper()}{param[1:]}", 0))
                    try:
                        params[param] = link[pos]
                    except IndexError:
                        raise HTTPException(403, "Forbidden")
                if params[param] not in link: raise HTTPException(403, "Forbidden")
            request.scope["path_params"] = params

        return depends(Depends(wrapper))
