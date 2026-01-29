import unittest

from sqlalchemy.ext.asyncio import create_async_engine

from SharedBackend.managers import *


class TestAPIKeyManager(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = create_async_engine('sqlite+aiosqlite:///:memory:')
        cls.manager = ApiKeyManager(cls.engine)

    async def asyncSetUp(self):
        await self.manager.init_db()

    async def test_generate_key(self):
        key = self.manager.generate_key()
        self.assertEqual(len(key) // 2, self.manager.UID_SIZE + self.manager.KEY_SIZE)

    async def test_register_verify_key(self):
        key = self.manager.generate_key()
        scopes = {"user:read", "user:write"}
        await self.manager.register_key(key, *scopes)
        api_scopes = await self.manager.verify_key(key)
        self.assertEqual(scopes.intersection(api_scopes), scopes)


if __name__ == '__main__':
    unittest.main()
