from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from SharedBackend.managers import BaseEncryptionManager


class TransactionSecurityMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, enc_manager: "BaseEncryptionManager"):
        super().__init__(app)
        self.enc_manager = enc_manager

    async def dispatch(self, request: "Request", call_next):
        # Decrypt incoming request
        encryptionId = request.headers.get("X-Transaction")
        if encryptionId:
            body = await request.body()
            decrypted_body = await self.enc_manager.decrypt(body.decode(), encryptionId)
            request._body = decrypted_body.encode()

        # Encrypt outgoing response
        response = await call_next(request)
        encryptionId = response.headers.get('X-Transaction')
        if response.status_code == 200 and encryptionId:
            response_body = b"".join([chunk async for chunk in response.body_iterator])
            encrypted_body = await self.enc_manager.encrypt(response_body.decode(), encryptionId)

            response = Response(
                content=encrypted_body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type=response.media_type,
            )
            response.headers["Content-Length"] = str(len(encrypted_body))

        return response
