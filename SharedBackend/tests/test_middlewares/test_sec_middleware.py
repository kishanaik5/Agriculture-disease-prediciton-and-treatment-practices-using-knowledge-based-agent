import json
import unittest
from unittest.mock import AsyncMock, MagicMock

from fastapi import Request, Response

from middlewares import TransactionSecurityMiddleware
from managers import AES256EncryptionManager


class TestTransactionSecurityMiddleware(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.enc_manager = MagicMock(spec=AES256EncryptionManager)
        self.middleware = TransactionSecurityMiddleware(app=MagicMock(), enc_manager=self.enc_manager)

    async def test_valid_encrypted_transaction(self):
        request1 = Request(
            scope={"type": "http", "headers": [(b"x-transaction", b"true")], "method": "POST", "path": "/"},
            receive=AsyncMock(return_value={"type": "http.request", "body": b"encrypted_body"})
        )
        request2 = Request(
            scope={"type": "http", "headers": [], "method": "POST", "path": "/"},
            receive=AsyncMock(return_value={"type": "http.request", "body": b"encrypted_body"})
        )

        async def call_next(request: "Request", enc=False):
            return Response(
                content=json.dumps({
                    "response": "valid_transaction",
                    "request": (await request.body()).decode()
                }),
                media_type="application/json",
                headers={"x-transaction": "true"} if enc else {}
            )

        self.enc_manager.decrypt.return_value = "decrypted_body"
        self.enc_manager.encrypt.return_value = "encrypted_body"
        response1 = await self.middleware.dispatch(request1, lambda req: call_next(req, enc=False))
        response2 = await self.middleware.dispatch(request2, lambda req: call_next(req, enc=True))

        self.assertEqual(response1.status_code, 200)
        self.assertEqual(response2.status_code, 200)
        self.assertEqual(json.loads(response1.body), {"response": "valid_transaction", "request": "decrypted_body"})
        self.assertEqual(response2.body, b"encrypted_body")


if __name__ == '__main__':
    unittest.main()
