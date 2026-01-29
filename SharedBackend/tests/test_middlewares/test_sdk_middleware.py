import unittest
from unittest.mock import AsyncMock, MagicMock

from starlette.requests import Request
from starlette.responses import Response

from SharedBackend.middlewares.SDKMiddleware import SDKMiddleware
from SharedBackend.managers import ApiKeyManager


class TestSDKMiddleware(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.api_key_manager = MagicMock(spec=ApiKeyManager)
        self.middleware = SDKMiddleware(app=MagicMock(), key_manager=self.api_key_manager)

    async def test_valid_api_key(self):
        self.api_key_manager.verify_key.return_value = {*self.api_key_manager.DEFAULT_SCOPES}
        request = Request(scope={"type": "http", "headers": [(b"x-api-key", b"valid_key")]})

        call_next = AsyncMock(return_value=Response(status_code=200))

        response = await self.middleware.dispatch(request, call_next)
        self.assertEqual(response.status_code, 200)
        call_next.assert_awaited_once()

    async def test_invalid_api_key(self):
        self.api_key_manager.verify_key.return_value = {}
        request = Request(scope={"type": "http", "headers": [(b"x-api-key", b"invalid_key")]})
        call_next = AsyncMock(return_value=Response(status_code=200))

        response = await self.middleware.dispatch(request, call_next)
        self.assertEqual(401, response.status_code)
        call_next.assert_not_awaited()

    async def test_missing_api_key(self):
        request = Request(scope={"type": "http", "headers": []})
        call_next = AsyncMock()

        response = await self.middleware.dispatch(request, call_next)
        self.assertEqual(401, response.status_code)
        call_next.assert_not_awaited()


if __name__ == '__main__':
    unittest.main()
