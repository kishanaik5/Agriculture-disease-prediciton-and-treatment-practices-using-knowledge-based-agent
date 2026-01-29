import jwt
from datetime import datetime, timedelta, UTC
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware


class JWTAuthMiddleware(BaseHTTPMiddleware):
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES = 30
    REFRESH_TOKEN_EXPIRE_DAYS = 7

    def __init__(self, app, jwt_secret: str):
        super().__init__(app)
        self.jwt_secret = jwt_secret

    def create_access_token(self, data: dict, expires_delta: timedelta = None):
        to_encode = data.copy()
        expire = datetime.now(UTC) + (
            expires_delta if expires_delta else timedelta(minutes=self.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm=self.ALGORITHM)
        return encoded_jwt

    def create_refresh_token(self, data: dict):
        return self.create_access_token(data, expires_delta=timedelta(days=self.REFRESH_TOKEN_EXPIRE_DAYS))

    def verify_jwt_token(self, token: str):
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            return HTTPException(status_code=402, detail="Token has expired")
        except jwt.InvalidTokenError:
            return HTTPException(status_code=401, detail="Invalid token")

    async def dispatch(self, request: Request, call_next):
        jwt_access_token = request.headers.get("X-ACCESS-JWT")
        jwt_refresh_token = request.headers.get("X-REFRESH-JWT")
        new_jwt_access_token = None

        if jwt_refresh_token:
            payload = self.verify_jwt_token(jwt_refresh_token)
            new_jwt_access_token = self.create_access_token(data=payload)
            upid = payload.get("uid")
        elif jwt_access_token:
            payload = self.verify_jwt_token(jwt_access_token)
            upid = payload.get("uid")
        else:
            upid = None

        request.state.upstreamId = upid
        response = await call_next(request)

        if response.headers.get("SET-X-JWT"):
            upid = response.headers.get("SET-X-JWT")
            del response.headers["SET-X-JWT"]
            response.headers["SET-X-ACCESS-JWT"] = self.create_access_token(data={"uid": upid})
            response.headers["SET-X-REFRESH-JWT"] = self.create_refresh_token(data={"uid": upid})
        elif new_jwt_access_token:
            response.headers["SET-X-ACCESS-JWT"] = new_jwt_access_token

        return response
